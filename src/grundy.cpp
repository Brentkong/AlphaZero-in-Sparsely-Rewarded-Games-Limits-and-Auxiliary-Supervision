#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
using namespace std;

vector<int> canonical_shape(const vector<vector<int>>& state) {
    vector<int> shape;
    for (const auto& row : state) {
        int cnt = 0;
        for (int cell : row) cnt += cell;
        shape.push_back(cnt);
    }
    return shape;
}


vector<vector<int>> chomp_children(const vector<int>& shape) {
    int rows = shape.size();
    vector<vector<int>> children;

    for (int r = 0; r < rows; r++) {
        int row_len = shape[r];

        for (int c = 0; c < row_len; c++) {

            if (r == 0 && c == 0) continue;

            vector<int> new_shape = shape;

            for (int rr = r; rr < rows; rr++) {
                new_shape[rr] = min(new_shape[rr], c);
            }

            children.push_back(new_shape);
        }
    }
    return children;
}


string shape_key(const vector<int>& v) {
    string s;
    for (int x : v) {
        s += to_string(x) + ",";
    }
    return s;
}


unordered_map<string, int> memo;


int grundy(const vector<int>& shape) {
    string key = shape_key(shape);
    if (memo.count(key)) return memo[key];

    vector<vector<int>> moves = chomp_children(shape);
    if (moves.empty()) return memo[key] = 0;

    unordered_set<int> gvals;
    for (const auto& child : moves) {
        gvals.insert(grundy(child));
    }

    int mex = 0;
    while (gvals.count(mex)) mex++;

    return memo[key] = mex;
}

int calculate_grundy(const vector<vector<int>>& state) {
    vector<int> shape = canonical_shape(state);
    return grundy(shape);
}


vector<pair<vector<int>, pair<int,int>>> find_optimal_moves(const vector<vector<int>>& state) {
    vector<int> shape = canonical_shape(state);
    int rows = shape.size();

    vector<pair<vector<int>, pair<int,int>>> optimal;

    for (int r = 0; r < rows; r++) {
        int row_len = shape[r];
        for (int c = 0; c < row_len; c++) {

            if (r == 0 && c == 0) continue;

            vector<int> new_shape = shape;
            for (int rr = r; rr < rows; rr++) {
                new_shape[rr] = min(new_shape[rr], c);
            }

            int g = grundy(new_shape);
            if (g == 0) {
                optimal.push_back({new_shape, {r, c}});
            }
        }
    }
    return optimal;
}


extern "C" {
    int grundy_value(int* flat_state, int rows, int cols) {
        vector<vector<int>> state(rows, vector<int>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                state[i][j] = flat_state[i * cols + j];

        return calculate_grundy(state);
    }

    
    int find_best_moves(
        int* flat_state,
        int rows,
        int cols,
        int* out_moves,
        int max_moves
    ) {
        vector<vector<int>> state(rows, vector<int>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                state[i][j] = flat_state[i * cols + j];

        auto optimal = find_optimal_moves(state);

        int count = std::min((int)optimal.size(), max_moves);

        for (int i = 0; i < count; i++) {
            out_moves[2 * i]     = optimal[i].second.first;  
            out_moves[2 * i + 1] = optimal[i].second.second; 
        }

        return count;  
    }
}