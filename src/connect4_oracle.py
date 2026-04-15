from __future__ import annotations

import atexit
import subprocess
import threading
from collections import OrderedDict, deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple, Union
from config import *
import numpy as np

SeqLike = Union[str, Sequence[int]]


class _LRUCache:

    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._data: "OrderedDict[str, List[int]]" = OrderedDict()

    def get(self, key: str) -> Optional[List[int]]:
        try:
            val = self._data.pop(key)
        except KeyError:
            return None
        self._data[key] = val
        return val

    def put(self, key: str, val: List[int]) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = val
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)


class PerfectC4Oracle:

    def __init__(
        self,
        game,
        solver_dir: Union[str, Path] = args['solver_path'],
        *,
        solver_exe: str = "c4solver",
        book_path: Optional[Union[str, Path]] = None,
        weak: bool = False,
        cache_size: int = 250_000,
        auto_start: bool = True,
    ):
        self.game = game
        self.solver_dir = Path(solver_dir).resolve()
        self.solver_path = str((self.solver_dir / solver_exe).resolve())
        self.book_path = None if book_path is None else str(Path(book_path).resolve())
        self.weak = bool(weak)

        if not Path(self.solver_path).is_file():
            raise FileNotFoundError(
                f"Could not find solver executable at: {self.solver_path}\n"
                "Make sure you built `c4solver` and passed the correct `solver_dir` "
                "(directory containing the binary)."
            )
        if self.book_path is not None and not Path(self.book_path).is_file():
            raise FileNotFoundError(
                f"Could not find opening book at: {self.book_path}\n"
                "Pass a valid `book_path=` or place `7x6.book` in `solver_dir`."
            )

        assert game.row_count == 6 and game.column_count == 7, (
            "This oracle wrapper expects a 7x6 solver. "
            "Rebuild the solver for other sizes (or extend this wrapper)."
        )

        self._proc: Optional[subprocess.Popen] = None
        self._io_lock = threading.Lock()

        self._stderr_lines: Deque[str] = deque(maxlen=200)
        self._stderr_thread: Optional[threading.Thread] = None

        self._cache = _LRUCache(cache_size) if cache_size and cache_size > 0 else None

        atexit.register(self.close)

        if auto_start:
            self.start()


    def start(self) -> None:

        if self._proc is not None and self._proc.poll() is None:
            return

        cmd: List[str] = [self.solver_path, "-a"]
        if self.weak:
            cmd.append("-w")
        if self.book_path is not None:
            cmd.extend(["-b", self.book_path])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.solver_dir),
            text=True,
            bufsize=1,  
        )
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        self._stderr_lines.clear()

        def _drain_stderr(proc: subprocess.Popen) -> None:
            try:
                for ln in proc.stderr:  
                    self._stderr_lines.append(ln.rstrip("\n"))
            except Exception:
                return

        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._proc,), daemon=True
        )
        self._stderr_thread.start()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()
        except Exception:
            pass

    def __enter__(self) -> "PerfectC4Oracle":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def history_to_seq(self, history: Sequence[int]) -> str:
        return "".join(str(int(c) + 1) for c in history)

    def _recent_stderr(self, n: int = 40) -> str:
        if not self._stderr_lines:
            return ""
        tail = list(self._stderr_lines)[-n:]
        return "\n".join(tail)

    def _parse_scores(self, seq: str, out_line: str) -> List[int]:
        C = int(self.game.column_count)
        parts = out_line.strip().split()
        if len(parts) < C:
            raise RuntimeError(
                "Solver output line was too short to contain scores.\n"
                f"seq={seq!r}\n"
                f"out_line={out_line!r}\n"
                f"recent_stderr:\n{self._recent_stderr()}"
            )

        tail = parts[-C:]
        try:
            scores = [int(x) for x in tail]
        except ValueError as e:
            raise RuntimeError(
                "Could not parse solver scores.\n"
                f"seq={seq!r}\n"
                f"out_line={out_line!r}\n"
                f"recent_stderr:\n{self._recent_stderr()}"
            ) from e

        if len(scores) != C:
            raise RuntimeError(
                "Solver returned unexpected number of scores.\n"
                f"seq={seq!r}\n"
                f"out_line={out_line!r}\n"
                f"scores={scores!r}\n"
                f"recent_stderr:\n{self._recent_stderr()}"
            )

        return scores

    def _query_many(self, seqs: List[str]) -> List[List[int]]:
        if not seqs:
            return []

        self.start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        payload = "\n".join(seqs) + "\n"  

        with self._io_lock:
            if self._proc.poll() is not None:
                self.close()
                self.start()
                assert self._proc is not None
                assert self._proc.stdin is not None
                assert self._proc.stdout is not None

            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except BrokenPipeError:
                self.close()
                self.start()
                assert self._proc is not None
                assert self._proc.stdin is not None
                assert self._proc.stdout is not None
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()

            out_lines: List[str] = []
            for _ in range(len(seqs)):
                ln = self._proc.stdout.readline()
                if ln == "":
                    raise RuntimeError(
                        "Solver process terminated unexpectedly while reading output.\n"
                        f"recent_stderr:\n{self._recent_stderr()}"
                    )
                out_lines.append(ln.rstrip("\n"))

        return [self._parse_scores(seq, ln) for seq, ln in zip(seqs, out_lines)]


    def analyze(self, history: SeqLike) -> List[int]:
        return self.analyze_batch([history])[0]

    def analyze_batch(self, histories: Sequence[SeqLike]) -> List[List[int]]:
        seqs: List[str] = [
            h if isinstance(h, str) else self.history_to_seq(h) for h in histories
        ]

        if self._cache is None:
            return self._query_many(seqs)

        out: List[Optional[List[int]]] = [None] * len(seqs)
        missing: List[str] = []
        missing_idx: List[int] = []

        for i, seq in enumerate(seqs):
            cached = self._cache.get(seq)
            if cached is not None:
                out[i] = cached
            else:
                missing.append(seq)
                missing_idx.append(i)

        if missing:
            solved = self._query_many(missing)
            for seq, scores in zip(missing, solved):
                self._cache.put(seq, scores)
            for i, scores in zip(missing_idx, solved):
                out[i] = scores

        if any(x is None for x in out):
            raise RuntimeError("Internal oracle error: missing results in batch")
        return [x for x in out]  

    def best_move(self, state: np.ndarray, history: Sequence[int]):
        last_action = history[-1] if history else None
        _, terminated = self.game.get_value_and_terminated(state, last_action)
        if terminated:
            return None

        valid = self.game.get_valid_moves(state).astype(bool)
        scores = self.analyze(history)

        C = self.game.column_count
        masked_scores = [scores[c] if valid[c] else -10**9 for c in range(C)]
        center_order = sorted(
            range(self.game.column_count),
            key=lambda c: abs(c - self.game.column_count // 2),
        )
        best_score = max(masked_scores)
        best_candidates = [c for c in center_order if masked_scores[c] == best_score]
        return best_candidates, scores

    def best_move_batch(
        self,
        states: np.ndarray,
        histories: Sequence[Sequence[int]],
    ) -> List[Optional[Tuple[List[int], List[int]]]]:
        if len(histories) == 0:
            return []
        assert len(states) == len(histories)

        last_actions = [h[-1] if h else None for h in histories]
        terminals = [
            self.game.get_value_and_terminated(s, a)[1]
            for s, a in zip(states, last_actions)
        ]

        idx = [i for i, t in enumerate(terminals) if not t]
        if not idx:
            return [None for _ in histories]

        scores_list = self.analyze_batch([histories[i] for i in idx])

        results: List[Optional[Tuple[List[int], List[int]]]] = [None] * len(histories)
        C = self.game.column_count
        center_order = sorted(range(C), key=lambda c: abs(c - C // 2))

        for j, i in enumerate(idx):
            scores = scores_list[j]
            valid = self.game.get_valid_moves(states[i]).astype(bool)
            masked_scores = [scores[c] if valid[c] else -10**9 for c in range(C)]
            best_score = max(masked_scores)
            best_candidates = [c for c in center_order if masked_scores[c] == best_score]
            results[i] = (best_candidates, scores)

        return results