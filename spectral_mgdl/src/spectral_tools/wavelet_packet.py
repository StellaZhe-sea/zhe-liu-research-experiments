"""
basis/wavelet_packet.py
T_wavelet_packet: optimal subband input encoding based on WPT of inv_L(e_n).

Physical meaning of s_n:
  - Each column = one wavelet packet basis function at optimal subband node
  - Optimal subband selected by Coifman-Wickerhauser algorithm (Shannon entropy)
  - Captures WHICH frequency band and WHERE in space the error lives

Natural advantage over DWT:
  - DWT only decomposes the approximation side — fixed frequency grouping
  - WPT decomposes both sides — finds the best frequency partition for THIS signal
  - When error has mixed frequency content (low + high simultaneously),
    WPT finds the right subband split; DWT cannot

Truncation: best basis algorithm selects optimal tree, then energy threshold on nodes.
inv_L: still exact DST (shared) — WPT only affects T_basis.
"""

import math
import numpy as np
import torch
import pywt
from typing import Tuple, List
from basis_mgdl.operators import HelmholtzOp


def _best_basis_node_costs(wp: pywt.WaveletPacket, max_level: int) -> dict:
    """
    Compute Shannon entropy cost for every node in the WP tree.
    Lower cost = more concentrated = better basis candidate.
    """
    costs = {}
    for level in range(max_level + 1):
        for node in wp.get_level(level, 'natural'):
            c = node.data
            p = c**2 / (np.sum(c**2) + 1e-12)
            p = p[p > 1e-12]
            costs[node.path] = float(-np.sum(p * np.log(p)))
    return costs


def _select_best_basis_nodes(wp: pywt.WaveletPacket, max_level: int) -> List[str]:
    """
    Coifman-Wickerhauser best basis: greedy bottom-up selection.
    For each parent, keep parent if cost(parent) <= cost(children sum).
    Returns list of node paths forming the optimal non-overlapping partition.
    """
    costs = _best_basis_node_costs(wp, max_level)

    def subtree_cost(path: str) -> float:
        """Minimum cost achievable by splitting this node."""
        if len(path) >= max_level:
            return costs.get(path, 1e12)
        child_a = path + 'a'
        child_d = path + 'd'
        if child_a not in costs or child_d not in costs:
            return costs.get(path, 1e12)
        split_cost = subtree_cost(child_a) + subtree_cost(child_d)
        return min(costs[path], split_cost)

    def best_nodes(path: str) -> List[str]:
        if len(path) >= max_level:
            return [path]
        child_a = path + 'a'
        child_d = path + 'd'
        if child_a not in costs or child_d not in costs:
            return [path]
        self_cost = costs.get(path, 1e12)
        split_cost = subtree_cost(child_a) + subtree_cost(child_d)
        if self_cost <= split_cost:
            return [path]
        return best_nodes(child_a) + best_nodes(child_d)

    return best_nodes('')


def _reconstruct_wp_basis_vector(
    wp: pywt.WaveletPacket, path: str, N: int
) -> np.ndarray:
    """
    Reconstruct a single WP basis vector in physical space.
    Zero out all nodes, set target node to its unit vector, reconstruct.
    """
    # Create a fresh WP with only this node active
    node_data = wp[path].data.copy()
    n_coeffs = len(node_data)

    cols = []
    for i in range(n_coeffs):
        # unit vector at position i in this node
        unit = np.zeros(n_coeffs)
        unit[i] = 1.0

        # Build a new WP tree for reconstruction
        wp_unit = pywt.WaveletPacket(
            data=None, wavelet=wp.wavelet, mode='periodization', maxlevel=wp.maxlevel
        )
        wp_unit[path] = unit

        rec = wp_unit.reconstruct(update=True)
        if len(rec) > N:
            rec = rec[:N]
        elif len(rec) < N:
            rec = np.pad(rec, (0, N - len(rec)))
        cols.append(rec)

    return np.stack(cols, axis=1)  # (N, n_coeffs)


def T_wavelet_packet(
    e_n: torch.Tensor,
    op: HelmholtzOp,
    wavelet: str = 'db4',
    max_level: int = None,
    energy_ratio: float = 0.99,
) -> Tuple[torch.Tensor, List[str], np.ndarray]:
    """
    Build wavelet packet input matrix s_n from residual e_n.

    Args:
        e_n:          residual L[u_n] - f,  shape (N,)
        op:           HelmholtzOp
        wavelet:      pywt wavelet name
        max_level:    WPT decomposition depth (default: log2(N)-2)
        energy_ratio: fraction of node energy to retain within best-basis nodes

    Returns:
        s_n:       (N, m) float32 — network input
        sel_nodes: list of (node_path, coeff_index) strings selected
        amps:      flattened amplitude array for logging
    """
    N = op.N
    if max_level is None:
        max_level = min(int(math.log2(N)) - 2, 4)  # cap at 4 for speed

    # delta = inv_L(e_n) = u_n - u*
    delta = op.inv_L(e_n).numpy()

    # Build WPT
    wp = pywt.WaveletPacket(
        data=delta, wavelet=wavelet, mode='periodization', maxlevel=max_level
    )

    # Find optimal subband partition
    best_nodes = _select_best_basis_nodes(wp, max_level)

    # Within each best-basis node, select coefficients by energy
    all_cols = []
    all_amps = []
    sel_info = []

    for node_path in best_nodes:
        node_coeffs = wp[node_path].data          # (n_k,)
        node_amps = np.abs(node_coeffs)
        all_amps.extend(node_amps.tolist())

        # Energy threshold within this node
        node_energy = np.sum(node_amps**2) + 1e-12
        sorted_idx = np.argsort(node_amps)[::-1]
        cumulative = 0.0
        sel_pos = []
        for idx in sorted_idx:
            sel_pos.append(idx)
            cumulative += node_amps[idx]**2
            if cumulative / node_energy >= energy_ratio:
                break

        if len(sel_pos) == 0:
            sel_pos = [int(np.argmax(node_amps))]

        a_sel = node_amps[sel_pos]
        a_max = a_sel.max() + 1e-12
        signs = np.sign(node_coeffs[sel_pos])

        for i, pos in enumerate(sel_pos):
            log_norm = math.log(1 + a_sel[i]) / math.log(1 + a_max)

            # Basis vector: unit at (node_path, pos)
            unit = np.zeros(len(node_coeffs))
            unit[pos] = 1.0
            wp_unit = pywt.WaveletPacket(
                data=None, wavelet=wavelet, mode='periodization', maxlevel=max_level
            )
            wp_unit[node_path] = unit
            basis_vec = wp_unit.reconstruct(update=True)
            if len(basis_vec) > N:
                basis_vec = basis_vec[:N]
            elif len(basis_vec) < N:
                basis_vec = np.pad(basis_vec, (0, N - len(basis_vec)))

            col = log_norm * signs[i] * basis_vec
            all_cols.append(col)
            sel_info.append(f"{node_path}[{pos}]")

    s_n = torch.tensor(np.stack(all_cols, axis=1), dtype=torch.float32)
    return s_n, sel_info, np.array(all_amps)
