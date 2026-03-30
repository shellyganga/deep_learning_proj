# Constrained interval cover problem

import numpy as np
import itertools
import random
from collections import deque


def max_sum_boundaries(values, coords, K, L, start=0, end=None):
    N = len(values)
    if end is None:
        end = coords[-1]
    coords_ext = np.concatenate(([start], coords, [end]))
    values_ext = np.concatenate(([0], values, [0]))
    dp = np.full((K+2, N+2), -np.inf, dtype=float)
    parent = np.full((K+2, N+2), -1, dtype=int)
    # Base case: 0 chosen, at start
    dp[0, 0] = 0
    # First choice: must satisfy start→coords[i] ≤ L
    for i in range(1, N+1):
        if coords_ext[i] - start <= L:
            dp[1, i] = values_ext[i]
            parent[1, i] = 0
    # Transitions
    for k in range(2, K+1):
        dq = deque()
        left = 0
        for i in range(1, N+1):
            while left < i and coords_ext[i] - coords_ext[left] > L:
                if dq and dq[0] == left:
                    dq.popleft()
                left += 1
            if dq and dp[k-1, dq[0]] != -np.inf:
                cand = dp[k-1, dq[0]] + values_ext[i]
                if cand > dp[k, i]:
                    dp[k, i] = cand
                    parent[k, i] = dq[0]
            while dq and dp[k-1, dq[-1]] <= dp[k-1, i]:
                dq.pop()
            dq.append(i)
    # Last gap check: coords[i]→end ≤ L
    best_sum, best_i = -np.inf, -1
    for i in range(1, N+1):
        if dp[K, i] != -np.inf and end - coords_ext[i] <= L:
            if dp[K, i] > best_sum:
                best_sum = dp[K, i]
                best_i = i
    if best_sum == -np.inf:
        return None, None
    # Reconstruct
    chosen = []
    k, i = K, best_i
    while k > 0 and i > 0:
        chosen.append(i-1)  # map back to values index
        i = parent[k, i]
        k -= 1
    chosen.reverse()
    return best_sum, chosen


def brute_force_boundaries(values, coords, K, L, start=0, end=None):
    N = len(values)
    if end is None:
        end = coords[-1]
    best_sum = -np.inf
    best_choice = None
    for comb in itertools.combinations(range(N), K):
        chosen_coords = coords[list(comb)]
        chosen_vals = values[list(comb)]
        all_coords = np.concatenate(([start], chosen_coords, [end]))
        gaps = np.diff(all_coords)
        if np.all(gaps <= L):
            total = chosen_vals.sum()
            if total > best_sum:
                best_sum = total
                best_choice = list(comb)
    if best_choice is None:
        return None, None
    return best_sum, best_choice


def stress_test(num_tests=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    for t in range(num_tests):
        N = random.randint(3, 8)        # keep small for brute force
        coords = np.sort(np.random.randint(0, 50, size=N))
        values = np.random.randint(1, 20, size=N)
        start, end = 0, coords[-1] + random.randint(5, 15)
        K = random.randint(1, min(N, 4))
        L = random.randint(3, 15)
        sum_dp, chosen_dp = max_sum_boundaries(values, coords, K, L, start, end)
        sum_bf, chosen_bf = brute_force_boundaries(values, coords, K, L, start, end)
        if (sum_dp is None) != (sum_bf is None):
            print("Mismatch: one found solution, other didn't")
            print("coords:", coords, "values:", values, "K:", K, "L:", L, "end:", end)
            return False
        if sum_dp is not None:
            if abs(sum_dp - sum_bf) > 1e-9:
                print("Mismatch in sums!")
                print("coords:", coords, "values:", values, "K:", K, "L:", L, "end:", end)
                print("DP:", sum_dp, chosen_dp)
                print("BF:", sum_bf, chosen_bf)
                return False
    print(f"All {num_tests} tests passed ✅")
    return True