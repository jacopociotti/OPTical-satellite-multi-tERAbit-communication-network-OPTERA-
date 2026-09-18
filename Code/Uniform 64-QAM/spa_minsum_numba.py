"""Scaled Min-Sum LDPC decoder accelerated with Numba.

The scalar interface decodes one component-code row.  The batch interfaces
perform exactly the same row operations for all ``m`` rows of one Staircase
component matrix A_j, while keeping the Sliding-Window control flow in the
original decoder classes unchanged.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _minsum_decode_inplace(
    llr: np.ndarray,
    check_ptr: np.ndarray,
    edge_var: np.ndarray,
    max_iter: int,
    alpha: float,
    v2c: np.ndarray,
    c2v: np.ndarray,
    sum_c2v: np.ndarray,
    post_llr: np.ndarray,
    decoded: np.ndarray,
) -> bool:
    """Run scaled Min-Sum using preallocated arrays."""
    n = llr.size
    n_checks = check_ptr.size - 1
    n_edges = edge_var.size

    for e in range(n_edges):
        v2c[e] = llr[edge_var[e]]
        c2v[e] = 0.0

    converged = False

    for _ in range(max_iter):
        for check in range(n_checks):
            start = check_ptr[check]
            end = check_ptr[check + 1]

            sign_product = 1.0
            min1 = np.inf
            min2 = np.inf
            min1_edge = -1

            for e in range(start, end):
                value = v2c[e]
                if value < 0.0:
                    sign_product = -sign_product

                magnitude = abs(value)
                if magnitude < min1:
                    min2 = min1
                    min1 = magnitude
                    min1_edge = e
                elif magnitude < min2:
                    min2 = magnitude

            for e in range(start, end):
                edge_sign = sign_product
                if v2c[e] < 0.0:
                    edge_sign = -edge_sign

                edge_magnitude = min2 if e == min1_edge else min1
                c2v[e] = alpha * edge_sign * edge_magnitude

        for variable in range(n):
            sum_c2v[variable] = 0.0

        for e in range(n_edges):
            sum_c2v[edge_var[e]] += c2v[e]

        for e in range(n_edges):
            variable = edge_var[e]
            v2c[e] = llr[variable] + sum_c2v[variable] - c2v[e]

        for variable in range(n):
            value = llr[variable] + sum_c2v[variable]
            post_llr[variable] = value
            decoded[variable] = 1 if value < 0.0 else 0

        converged = True
        for check in range(n_checks):
            parity = 0
            start = check_ptr[check]
            end = check_ptr[check + 1]

            for e in range(start, end):
                parity ^= decoded[edge_var[e]]

            if parity != 0:
                converged = False
                break

        if converged:
            break

    return converged


@njit(cache=True, parallel=True)
def _decode_component_old_batch(
    L_I_prev: np.ndarray,
    L_E_prev: np.ndarray,
    L_I_curr: np.ndarray,
    L_E_curr: np.ndarray,
    raw_to_sys: np.ndarray,
    sys_to_raw: np.ndarray,
    check_ptr: np.ndarray,
    edge_var: np.ndarray,
    max_iter: int,
    minsum_alpha: float,
    alpha_c: float,
    alpha_n: float,
    track_stats: bool,
    llr_ws: np.ndarray,
    v2c_ws: np.ndarray,
    c2v_ws: np.ndarray,
    sum_ws: np.ndarray,
    post_ws: np.ndarray,
    decoded_ws: np.ndarray,
) -> tuple[int, float]:
    """Decode all rows of A_j for the fixed-window decoder.

    This is the original ``for k in range(m)`` loop, compiled and parallelized.
    Every row uses private work arrays and writes only its own column/row.
    """
    m = L_I_curr.shape[0]
    n = 2 * m
    converged_rows = 0
    row_mean_sum = 0.0

    for k in prange(m):
        llr = llr_ws[k]

        # Original construction:
        # row_sys = [ (L_I_prev + L_E_prev)[:, k],
        #             (L_I_curr + L_E_curr)[k, :] ]
        # row_raw = row_sys[raw_to_sys]
        for raw_idx in range(n):
            sys_idx = raw_to_sys[raw_idx]
            if sys_idx < m:
                llr[raw_idx] = L_I_prev[sys_idx, k] + L_E_prev[sys_idx, k]
            else:
                col = sys_idx - m
                llr[raw_idx] = L_I_curr[k, col] + L_E_curr[k, col]

        converged = _minsum_decode_inplace(
            llr,
            check_ptr,
            edge_var,
            max_iter,
            minsum_alpha,
            v2c_ws[k],
            c2v_ws[k],
            sum_ws[k],
            post_ws[k],
            decoded_ws[k],
        )

        if converged:
            converged_rows += 1
            scale = alpha_c
        else:
            scale = alpha_n

        abs_sum = 0.0

        # First systematic half: A_j -> B_{j-1}; write column k.
        for sys_idx in range(m):
            raw_idx = sys_to_raw[sys_idx]
            input_value = L_I_prev[sys_idx, k] + L_E_prev[sys_idx, k]
            ext = post_ws[k, raw_idx] - input_value
            if track_stats:
                abs_sum += abs(ext)
            L_E_prev[sys_idx, k] = scale * ext

        # Second systematic half: A_j -> B_j; write row k.
        for col in range(m):
            sys_idx = m + col
            raw_idx = sys_to_raw[sys_idx]
            input_value = L_I_curr[k, col] + L_E_curr[k, col]
            ext = post_ws[k, raw_idx] - input_value
            if track_stats:
                abs_sum += abs(ext)
            L_E_curr[k, col] = scale * ext

        if track_stats:
            row_mean_sum += abs_sum / n

    return converged_rows, row_mean_sum


@njit(cache=True, parallel=True)
def _decode_component_adaptive_batch(
    L_I_prev: np.ndarray,
    L_E_left_prev: np.ndarray,
    L_E_right_prev: np.ndarray,
    L_I_curr: np.ndarray,
    L_E_left_curr: np.ndarray,
    L_E_right_curr: np.ndarray,
    raw_to_sys: np.ndarray,
    sys_to_raw: np.ndarray,
    check_ptr: np.ndarray,
    edge_var: np.ndarray,
    max_iter: int,
    minsum_alpha: float,
    alpha_c: float,
    alpha_n: float,
    track_stats: bool,
    llr_ws: np.ndarray,
    v2c_ws: np.ndarray,
    c2v_ws: np.ndarray,
    sum_ws: np.ndarray,
    post_ws: np.ndarray,
    decoded_ws: np.ndarray,
) -> tuple[int, float]:
    """Decode all rows of A_j for the adaptive decoder.

    The input/output memories are exactly those of the original implementation:
    input from ``L_E_left[j-1]`` and ``L_E_right[j]``; output to
    ``L_E_right[j-1]`` and ``L_E_left[j]``.
    """
    m = L_I_curr.shape[0]
    n = 2 * m
    converged_rows = 0
    row_mean_sum = 0.0

    for k in prange(m):
        llr = llr_ws[k]

        for raw_idx in range(n):
            sys_idx = raw_to_sys[raw_idx]
            if sys_idx < m:
                llr[raw_idx] = (
                    L_I_prev[sys_idx, k] + L_E_left_prev[sys_idx, k]
                )
            else:
                col = sys_idx - m
                llr[raw_idx] = L_I_curr[k, col] + L_E_right_curr[k, col]

        converged = _minsum_decode_inplace(
            llr,
            check_ptr,
            edge_var,
            max_iter,
            minsum_alpha,
            v2c_ws[k],
            c2v_ws[k],
            sum_ws[k],
            post_ws[k],
            decoded_ws[k],
        )

        if converged:
            converged_rows += 1
            scale = alpha_c
        else:
            scale = alpha_n

        abs_sum = 0.0

        # Message A_j -> B_{j-1}: destination L_E_right[j-1].
        for sys_idx in range(m):
            raw_idx = sys_to_raw[sys_idx]
            input_value = np.float32(
                L_I_prev[sys_idx, k] + L_E_left_prev[sys_idx, k]
            )
            ext = np.float32(post_ws[k, raw_idx] - input_value)
            if track_stats:
                abs_sum += abs(ext)
            L_E_right_prev[sys_idx, k] = np.float32(scale * ext)

        # Message A_j -> B_j: destination L_E_left[j].
        for col in range(m):
            sys_idx = m + col
            raw_idx = sys_to_raw[sys_idx]
            input_value = np.float32(
                L_I_curr[k, col] + L_E_right_curr[k, col]
            )
            ext = np.float32(post_ws[k, raw_idx] - input_value)
            if track_stats:
                abs_sum += abs(ext)
            L_E_left_curr[k, col] = np.float32(scale * ext)

        if track_stats:
            row_mean_sum += abs_sum / n

    return converged_rows, row_mean_sum


@njit(cache=True)
def _count_staircase_syndrome(
    B_prev: np.ndarray,
    B_curr: np.ndarray,
    raw_to_sys: np.ndarray,
    check_ptr: np.ndarray,
    edge_var: np.ndarray,
) -> int:
    m = B_curr.shape[0]
    n_checks = check_ptr.size - 1
    unsatisfied = 0

    for row in range(m):
        for check in range(n_checks):
            parity = 0
            start = check_ptr[check]
            end = check_ptr[check + 1]

            for e in range(start, end):
                raw_variable = edge_var[e]
                sys_variable = raw_to_sys[raw_variable]

                if sys_variable < m:
                    bit = B_prev[sys_variable, row]
                else:
                    bit = B_curr[row, sys_variable - m]

                parity ^= bit

            unsatisfied += parity

    return unsatisfied


def _build_check_adjacency(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(H, "tocsr"):
        H_csr = H.tocsr()
        H_csr.sort_indices()
        check_ptr = np.asarray(H_csr.indptr, dtype=np.int32)
        edge_var = np.asarray(H_csr.indices, dtype=np.int32)
    else:
        H_dense = np.asarray(H)
        rows, cols = np.nonzero(H_dense)
        n_checks = H_dense.shape[0]
        counts = np.bincount(rows, minlength=n_checks)
        check_ptr = np.empty(n_checks + 1, dtype=np.int32)
        check_ptr[0] = 0
        np.cumsum(counts, out=check_ptr[1:])
        edge_var = np.asarray(cols, dtype=np.int32)

    return np.ascontiguousarray(check_ptr), np.ascontiguousarray(edge_var)


class NumbaMinSumDecoder:
    """Reusable Numba-accelerated scaled Min-Sum component decoder."""

    def __init__(self, H: np.ndarray, alpha: float = 0.75, dtype=np.float32):
        self.check_ptr, self.edge_var = _build_check_adjacency(H)
        self.alpha = float(alpha)
        self.dtype = np.dtype(dtype)

        self.n = int(H.shape[1])
        self.n_edges = int(self.edge_var.size)

        # Workspace for the scalar compatibility interface.
        self.v2c = np.empty(self.n_edges, dtype=self.dtype)
        self.c2v = np.empty(self.n_edges, dtype=self.dtype)
        self.sum_c2v = np.empty(self.n, dtype=self.dtype)
        self.post_llr = np.empty(self.n, dtype=self.dtype)
        self.decoded = np.empty(self.n, dtype=np.uint8)

        warmup_llr = np.ones(self.n, dtype=self.dtype)
        _minsum_decode_inplace(
            warmup_llr,
            self.check_ptr,
            self.edge_var,
            1,
            self.alpha,
            self.v2c,
            self.c2v,
            self.sum_c2v,
            self.post_llr,
            self.decoded,
        )

        self.identity_raw_to_sys = np.arange(self.n, dtype=np.int64)
        self.identity_sys_to_raw = np.arange(self.n, dtype=np.int64)
        self.batch_m = 0

    def prepare_batch(self, m: int) -> None:
        """Allocate one private workspace per Staircase row."""
        m = int(m)
        if self.batch_m == m:
            return
        if 2 * m != self.n:
            raise ValueError(f"Expected 2*m == {self.n}, got m={m}.")

        self.batch_m = m
        self.llr_batch = np.empty((m, self.n), dtype=self.dtype)
        self.v2c_batch = np.empty((m, self.n_edges), dtype=self.dtype)
        self.c2v_batch = np.empty((m, self.n_edges), dtype=self.dtype)
        self.sum_batch = np.empty((m, self.n), dtype=self.dtype)
        self.post_batch = np.empty((m, self.n), dtype=self.dtype)
        self.decoded_batch = np.empty((m, self.n), dtype=np.uint8)

    def __call__(self, llr, H_unused=None, max_iter: int = 10):
        llr_array = np.asarray(llr, dtype=self.dtype)
        if llr_array.ndim != 1 or llr_array.size != self.n:
            raise ValueError(
                f"Expected a one-dimensional LLR vector of length {self.n}, "
                f"got shape {llr_array.shape}."
            )
        if not llr_array.flags.c_contiguous:
            llr_array = np.ascontiguousarray(llr_array)

        converged = _minsum_decode_inplace(
            llr_array,
            self.check_ptr,
            self.edge_var,
            int(max_iter),
            self.alpha,
            self.v2c,
            self.c2v,
            self.sum_c2v,
            self.post_llr,
            self.decoded,
        )
        return self.post_llr, bool(converged)

    def decode_component_old(
        self,
        L_I_prev,
        L_E_prev,
        L_I_curr,
        L_E_curr,
        raw_to_sys,
        sys_to_raw,
        max_iter,
        alpha_c,
        alpha_n,
        track_stats=False,
    ):
        self.prepare_batch(L_I_curr.shape[0])
        raw_to_sys_array = (
            self.identity_raw_to_sys
            if raw_to_sys is None
            else np.asarray(raw_to_sys, dtype=np.int64)
        )
        sys_to_raw_array = (
            self.identity_sys_to_raw
            if sys_to_raw is None
            else np.asarray(sys_to_raw, dtype=np.int64)
        )
        return _decode_component_old_batch(
            L_I_prev,
            L_E_prev,
            L_I_curr,
            L_E_curr,
            raw_to_sys_array,
            sys_to_raw_array,
            self.check_ptr,
            self.edge_var,
            int(max_iter),
            self.alpha,
            float(alpha_c),
            float(alpha_n),
            bool(track_stats),
            self.llr_batch,
            self.v2c_batch,
            self.c2v_batch,
            self.sum_batch,
            self.post_batch,
            self.decoded_batch,
        )

    def decode_component_adaptive(
        self,
        L_I_prev,
        L_E_left_prev,
        L_E_right_prev,
        L_I_curr,
        L_E_left_curr,
        L_E_right_curr,
        raw_to_sys,
        sys_to_raw,
        max_iter,
        alpha_c,
        alpha_n,
        track_stats=False,
    ):
        self.prepare_batch(L_I_curr.shape[0])
        raw_to_sys_array = (
            self.identity_raw_to_sys
            if raw_to_sys is None
            else np.asarray(raw_to_sys, dtype=np.int64)
        )
        sys_to_raw_array = (
            self.identity_sys_to_raw
            if sys_to_raw is None
            else np.asarray(sys_to_raw, dtype=np.int64)
        )
        return _decode_component_adaptive_batch(
            L_I_prev,
            L_E_left_prev,
            L_E_right_prev,
            L_I_curr,
            L_E_left_curr,
            L_E_right_curr,
            raw_to_sys_array,
            sys_to_raw_array,
            self.check_ptr,
            self.edge_var,
            int(max_iter),
            self.alpha,
            float(alpha_c),
            float(alpha_n),
            bool(track_stats),
            self.llr_batch,
            self.v2c_batch,
            self.c2v_batch,
            self.sum_batch,
            self.post_batch,
            self.decoded_batch,
        )

    def warmup_staircase_kernels(self, m: int, perm=None) -> None:
        """Compile both batch kernels before simulation timing starts."""
        self.prepare_batch(m)
        if perm is None:
            sys_to_raw = self.identity_sys_to_raw
            raw_to_sys = self.identity_raw_to_sys
        else:
            sys_to_raw = np.asarray(perm, dtype=np.int64)
            raw_to_sys = np.empty_like(sys_to_raw)
            raw_to_sys[sys_to_raw] = np.arange(sys_to_raw.size, dtype=np.int64)

        L_I_prev = np.ones((m, m), dtype=np.float32)
        L_I_curr = np.ones((m, m), dtype=np.float32)

        # Fixed-window decoder uses float32 extrinsic memories.
        L_E_prev = np.zeros((m, m), dtype=np.float32)
        L_E_curr = np.zeros((m, m), dtype=np.float32)
        self.decode_component_old(
            L_I_prev, L_E_prev, L_I_curr, L_E_curr,
            raw_to_sys, sys_to_raw, 1, 0.75, 0.375, False,
        )

        # Preserve the current adaptive-decoder dtype (float64 memories).
        L_E_left_prev = np.zeros((m, m), dtype=np.float64)
        L_E_right_prev = np.zeros((m, m), dtype=np.float64)
        L_E_left_curr = np.zeros((m, m), dtype=np.float64)
        L_E_right_curr = np.zeros((m, m), dtype=np.float64)
        self.decode_component_adaptive(
            L_I_prev,
            L_E_left_prev,
            L_E_right_prev,
            L_I_curr,
            L_E_left_curr,
            L_E_right_curr,
            raw_to_sys,
            sys_to_raw,
            1,
            0.75,
            0.375,
            False,
        )

    def count_staircase_syndrome(self, B_prev, B_curr, raw_to_sys=None):
        B_prev_array = np.asarray(B_prev, dtype=np.uint8)
        B_curr_array = np.asarray(B_curr, dtype=np.uint8)
        mapping = (
            self.identity_raw_to_sys
            if raw_to_sys is None
            else np.asarray(raw_to_sys, dtype=np.int64)
        )
        return int(
            _count_staircase_syndrome(
                B_prev_array,
                B_curr_array,
                mapping,
                self.check_ptr,
                self.edge_var,
            )
        )
