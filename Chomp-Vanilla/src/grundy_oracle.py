import ctypes
import numpy as np
from config import args

lib = ctypes.CDLL("./libgrundy.dylib")

lib.grundy_value.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int
]
lib.grundy_value.restype = ctypes.c_int

lib.find_best_moves.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.find_best_moves.restype = ctypes.c_int



def flatten_state(state):
    return state.astype(np.int32).flatten()

def fits_grundy_limit(state, oracle_size):
    return int(np.sum(state > 0.5)) <= oracle_size

def grundy(state):
    flat = flatten_state(state)
    return lib.grundy_value(
         flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
         state.shape[0],
         state.shape[1]
    )

def find_best_moves(state, max_moves=64):
    rows, cols = state.shape
    flat = flatten_state(state)

    out_moves = (ctypes.c_int * (2 * max_moves))()

    count = lib.find_best_moves(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rows,
        cols,
        out_moves,
        max_moves,
    )

    if count == 0:
        return []

    moves = []
    for i in range(count):
        r = out_moves[2 * i]
        c = out_moves[2 * i + 1]
        moves.append(r * cols + c)

    return moves

