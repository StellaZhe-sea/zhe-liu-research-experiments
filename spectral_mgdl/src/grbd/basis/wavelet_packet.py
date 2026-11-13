"""
grbd.basis.wavelet_packet
=========================
Wavelet Packet Transform (WPT) basis — Suite 2.

Unlike DWT (which only recurses on the approximation subband),
WPT builds a full binary tree and selects the "best basis" node
partition that minimises a cost function (entropy by default).

This gives a fully adaptive time-frequency tiling: the basis
automatically concentrates atoms where the residual energy lives,
rather than being forced into dyadic octave bands.

Best-basis criterion: Shannon entropy  H = -Σ p log p
Alternative:         log-energy,  ℓ¹ norm  (configurable)

Requirements: PyWavelets ≥ 1.4
"""

import numpy as np
from .base import BasisSolver

try:
    import pywt
    _HAS_PWT = True
except ImportError:
    _HAS_PWT = False


def _gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    if s[0] < 1e-30:
        return 1e-30
    mad = np.median(np.abs(s - np.median(s)))
    sig = mad / 0.6745 + 1e-12
    n   = len(amps); beta = 1.0 / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(max(omega * sig, floor_ratio * s[0]))


def _node_cost(coeffs: np.ndarray, criterion: str = "entropy") -> float:
    """Cost function for best-basis selection."""
    c = coeffs[coeffs != 0]
    if len(c) == 0:
        return 0.0
    if criterion == "entropy":
        e  = c ** 2; e /= (e.sum() + 1e-30)
        return float(-np.sum(e * np.log(e + 1e-30)))
    elif criterion == "log_energy":
        return float(-np.sum(np.log(c**2 + 1e-30)))
    elif criterion == "l1":
        return float(np.sum(np.abs(c)))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


class WaveletPacketBasis(BasisSolver):
    """
    Parameters
    ----------
    op            : PDEOperator
    family        : str           e.g. 'db4', 'sym6', 'coif3'
    levels        : int           max tree depth
    mode          : str           PyWavelets extension mode
    criterion     : str           best-basis cost: 'entropy'|'log_energy'|'l1'
    gd_floor      : float
    sobolev_alpha : float         frequency-band Sobolev weight exponent
    sobolev_tau   : float
    """

    def __init__(self, op, family='db4', levels=4, mode='symmetric',
                 criterion='entropy', gd_floor=0.10,
                 sobolev_alpha=1.0, sobolev_tau=1.0):
        if not _HAS_PWT:
            raise ImportError("PyWavelets required: pip install PyWavelets")
        self.op            = op
        self._N            = op.N
        self.wavelet       = pywt.Wavelet(family)
        self.levels        = levels
        self.mode          = mode
        self.criterion     = criterion
        self.gd_floor      = gd_floor
        self.sobolev_alpha = sobolev_alpha
        self.sobolev_tau   = sobolev_tau

    # ── best-basis tree search ───────────────────────────────────────────

    def _best_basis(self, u: np.ndarray) -> tuple:
        """
        Compute full WP tree, select best-basis node partition.

        Returns
        -------
        nodes       : list of (level, node_idx, coeffs)  — selected leaf nodes
        wp          : pywt.WaveletPacket object
        """
        wp    = pywt.WaveletPacket(data=u.astype(np.float64),
                                   wavelet=self.wavelet,
                                   mode=self.mode,
                                   maxlevel=self.levels)
        nodes = self._best_partition(wp, node_path='')
        return nodes, wp

    def _best_partition(self, wp, node_path: str) -> list:
        """
        Recursive best-basis selection via Coifman-Wickerhauser algorithm.
        """
        node = wp[node_path] if node_path else wp['']
        level = len(node_path)

        if level >= self.levels:
            return [(node_path, np.array(node.data, dtype=np.float32))]

        # Children
        path_a = node_path + 'a'
        path_d = node_path + 'd'
        children_a = self._best_partition(wp, path_a)
        children_d = self._best_partition(wp, path_d)

        # Cost of parent node vs sum of children costs
        cost_parent   = _node_cost(np.array(node.data), self.criterion)
        cost_children = sum(_node_cost(c, self.criterion)
                            for _, c in children_a + children_d)

        if cost_parent <= cost_children:
            return [(node_path, np.array(node.data, dtype=np.float32))]
        else:
            return children_a + children_d

    # ── reconstruct from best-basis node list ────────────────────────────

    def _reconstruct_from_nodes(self, nodes: list, wp_template) -> np.ndarray:
        """Reconstruct physical field from (possibly modified) node list."""
        wp2 = pywt.WaveletPacket(data=None, wavelet=self.wavelet,
                                 mode=self.mode, maxlevel=self.levels)
        for path, coeffs in nodes:
            wp2[path] = coeffs.astype(np.float64)
        rec = wp2.reconstruct(update=True)
        return rec[:self._N].astype(np.float32)

    # ── BasisSolver interface ────────────────────────────────────────────

    def decompose(self, residual: np.ndarray) -> tuple:
        nodes, wp    = self._best_basis(residual)

        # GD selection within best-basis nodes
        all_coeffs   = np.concatenate([c for _, c in nodes])
        amps         = np.abs(all_coeffs)
        tau          = _gd_threshold(amps, self.gd_floor)
        mask_flat    = amps > tau
        if not mask_flat.any():
            mask_flat[np.argmax(amps)] = True

        # per-node masks and slices
        node_slices  = []
        pos          = 0
        for path, c in nodes:
            n = len(c)
            node_slices.append((path, pos, pos + n))
            pos += n

        selected_nodes = []
        for (path, s, e) in node_slices:
            sel = mask_flat[s:e]
            if sel.any():
                selected_nodes.append((path, s, e, sel))

        coeffs = all_coeffs[mask_flat]
        meta   = dict(
            nodes=nodes, node_slices=node_slices,
            selected_nodes=selected_nodes,
            mask_flat=mask_flat, all_coeffs=all_coeffs,
            tau=tau, amps_res=amps,
            n_atoms=int(mask_flat.sum()),
            wp_wavelet=self.wavelet.name,
            wp_mode=self.mode,
        )
        return coeffs, meta

    def reconstruct(self, coeffs: np.ndarray, meta: dict) -> np.ndarray:
        full = meta["all_coeffs"].copy()
        full[meta["mask_flat"]] = coeffs
        # rebuild node list with updated coefficients
        pos      = 0
        new_nodes = []
        for path, c_orig in meta["nodes"]:
            n = len(c_orig)
            new_nodes.append((path, full[pos:pos+n].copy()))
            pos += n
        # reconstruct using a fresh WP object
        wp_temp = pywt.WaveletPacket(data=None, wavelet=self.wavelet,
                                     mode=self.mode, maxlevel=self.levels)
        for path, c in new_nodes:
            wp_temp[path] = c.astype(np.float64)
        rec = wp_temp.reconstruct(update=True)
        return rec[:self._N].astype(np.float32)

    def atom_features(self, residual: np.ndarray, target: np.ndarray,
                      meta: dict) -> tuple:
        # Decompose target on the same best-basis nodes
        wp_t     = pywt.WaveletPacket(data=target.astype(np.float64),
                                      wavelet=self.wavelet,
                                      mode=self.mode, maxlevel=self.levels)
        t_coeffs = np.concatenate([
            np.array(wp_t[path].data, dtype=np.float32)
            for path, _, _ in meta["node_slices"]
        ])
        tc       = t_coeffs[meta["mask_flat"]]
        a_sel    = np.abs(tc)
        signs    = np.sign(tc); signs[signs == 0] = 1.0

        alphas   = self.sobolev_weights({**meta, "target_coeffs": tc})
        H        = self._spectral_entropy(alphas)
        B        = self._build_atom_basis(meta)   # (N, M)
        s        = self._log_norm_encode(a_sel, signs, B)

        meta = {
            **meta,
            "target_coeffs": tc,
            "amps_delta":    a_sel,
            "alphas":        alphas,
            "entropy":       H,
            "target":        target,
            "s_input":       s,
        }
        return s, meta

    def _build_atom_basis(self, meta: dict) -> np.ndarray:
        """Build (N, M) spatial atom profile matrix."""
        M        = meta["n_atoms"]
        B        = np.zeros((self._N, M), dtype=np.float32)
        atom_idx = 0
        pos      = 0
        for path, c_orig in meta["nodes"]:
            n     = len(c_orig)
            s, e  = pos, pos + n
            for k in range(n):
                if meta["mask_flat"][s + k]:
                    indicator          = np.zeros(n, dtype=np.float64)
                    indicator[k]       = 1.0
                    wp_ind             = pywt.WaveletPacket(
                        data=None, wavelet=self.wavelet,
                        mode=self.mode, maxlevel=self.levels)
                    wp_ind[path]       = indicator
                    rec                = wp_ind.reconstruct(update=True)
                    B[:, atom_idx]     = rec[:self._N].astype(np.float32)
                    atom_idx          += 1
            pos += n
        return B

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        """
        WP Sobolev weights: use centre frequency of each selected node's
        subband as a proxy for 'wavenumber', then apply (τ²+ω²)^{-α/2}.
        """
        tc = meta.get("target_coeffs", None)
        if tc is None:
            return np.ones(meta["n_atoms"]) / meta["n_atoms"]

        if self.sobolev_alpha > 0:
            # Centre frequency for each atom's node
            omega_list = []
            for path, s, e in meta["node_slices"]:
                level = len(path)
                node_idx = sum(
                    (1 if c == 'd' else 0) * 2**(level - i - 1)
                    for i, c in enumerate(path)
                ) if path else 0
                # Centre freq ≈ (node_idx + 0.5) / 2^level × Nyquist
                n_nodes = 2 ** level
                freq    = (node_idx + 0.5) / n_nodes * 0.5   # normalised [0,0.5]
                n_sel   = int(meta["mask_flat"][s:e].sum())
                omega_list.extend([freq * self._N * 2 * np.pi] * n_sel)

            omega = np.array(omega_list[:len(tc)], dtype=np.float64)
            w     = (self.sobolev_tau**2 + omega**2) ** (-self.sobolev_alpha / 2.0)
            w     = w * tc**2
        else:
            w = tc**2

        w = np.abs(w)
        return w / (w.sum() + 1e-12)
