"""Validated temporal trajectories for the periodic Burgers paper data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch


@dataclass(frozen=True)
class TrajectorySpec:
    resolution: int = 256
    viscosity: float = 0.1
    domain_length: float = 1.0
    final_time: float = 1.0
    record_steps: int = 10
    solver_substeps_per_record: int = 20

    @property
    def solver_steps(self) -> int:
        return self.record_steps * self.solver_substeps_per_record

    @property
    def time_step(self) -> float:
        return self.final_time / self.solver_steps

    def validate(self) -> None:
        if 8192 % self.resolution:
            raise ValueError("Resolution must divide the native 8192-point grid")
        if self.viscosity <= 0.0 or self.final_time <= 0.0 or self.domain_length <= 0.0:
            raise ValueError("Viscosity, final time, and domain length must be positive")
        if self.record_steps <= 0 or self.solver_substeps_per_record <= 0:
            raise ValueError("Temporal step counts must be positive")


class BurgersETDRK4:
    """Batched periodic pseudo-spectral ETDRK4 Burgers solver."""

    def __init__(self, spec: TrajectorySpec, device: torch.device) -> None:
        spec.validate()
        self.spec = spec
        self.device = device
        n_grid = spec.resolution
        real_dtype = torch.float64
        frequency = torch.fft.fftfreq(
            n_grid, d=1.0 / n_grid, device=device, dtype=real_dtype
        )
        wave = 2.0 * torch.pi * frequency / spec.domain_length
        linear = -spec.viscosity * wave.square()
        dt = spec.time_step
        self.wave = wave
        self.dealias = frequency.abs() <= n_grid / 3.0
        self.e = torch.exp(dt * linear)
        self.e2 = torch.exp(0.5 * dt * linear)
        contour_count = 32
        roots = torch.exp(
            1j
            * torch.pi
            * (torch.arange(contour_count, device=device, dtype=real_dtype) + 0.5)
            / contour_count
        )
        lr = dt * linear[:, None].to(torch.complex128) + roots[None, :]
        self.q = dt * ((torch.exp(lr / 2.0) - 1.0) / lr).mean(1).real
        self.f1 = dt * (
            (-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr.square()))
            / lr.pow(3)
        ).mean(1).real
        self.f2 = dt * (
            (2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr.pow(3)
        ).mean(1).real
        self.f3 = dt * (
            (-4.0 - 3.0 * lr - lr.square() + torch.exp(lr) * (4.0 - lr))
            / lr.pow(3)
        ).mean(1).real

    def nonlinear(self, spectrum: torch.Tensor) -> torch.Tensor:
        physical = torch.fft.ifft(spectrum, dim=-1).real
        nonlinear = -0.5j * self.wave * torch.fft.fft(physical.square(), dim=-1)
        return nonlinear * self.dealias

    def step(self, spectrum: torch.Tensor) -> torch.Tensor:
        nv = self.nonlinear(spectrum)
        a = self.e2 * spectrum + self.q * nv
        na = self.nonlinear(a)
        b = self.e2 * spectrum + self.q * na
        nb = self.nonlinear(b)
        c = self.e2 * a + self.q * (2.0 * nb - nv)
        nc = self.nonlinear(c)
        return self.e * spectrum + self.f1 * nv + 2.0 * self.f2 * (na + nb) + self.f3 * nc

    @torch.no_grad()
    def solve(self, initial: torch.Tensor) -> torch.Tensor:
        if initial.ndim != 2 or initial.shape[-1] != self.spec.resolution:
            raise ValueError("Initial states must have shape (batch, resolution)")
        spectrum = torch.fft.fft(initial.to(self.device, torch.float64), dim=-1)
        records = [initial.to(self.device, torch.float64)]
        for step in range(1, self.spec.solver_steps + 1):
            spectrum = self.step(spectrum)
            if step % self.spec.solver_substeps_per_record == 0:
                records.append(torch.fft.ifft(spectrum, dim=-1).real)
        return torch.stack(records, dim=1).to(torch.float32)


def load_paper_fields(path: Path, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(path)
    stride = 8192 // resolution
    initial = np.asarray(data["a"][:, ::stride], dtype=np.float32)
    final = np.asarray(data["u"][:, ::stride], dtype=np.float32)
    return initial, final


@torch.no_grad()
def validate_terminal_state(
    data_path: Path,
    spec: TrajectorySpec,
    device: torch.device,
    samples: int = 4,
) -> dict[str, float]:
    initial, target = load_paper_fields(data_path, spec.resolution)
    solver = BurgersETDRK4(spec, device)
    trajectory = solver.solve(torch.from_numpy(initial[:samples]))
    prediction = trajectory[:, -1].cpu()
    target_tensor = torch.from_numpy(target[:samples])
    relative = (
        (prediction - target_tensor).flatten(1).norm(dim=1)
        / target_tensor.flatten(1).norm(dim=1).clamp_min(1.0e-12)
    )
    mean_error = (prediction.mean(-1) - target_tensor.mean(-1)).abs()
    return {
        "mean_relative_l2": float(relative.mean()),
        "max_relative_l2": float(relative.max()),
        "mean_conservation_error": float(mean_error.mean()),
    }


@torch.no_grad()
def generate_cached_trajectories(
    data_path: Path,
    cache_path: Path,
    spec: TrajectorySpec,
    device: torch.device,
    count: int,
    batch_size: int,
) -> torch.Tensor:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["spec"] != spec.__dict__ or int(payload["count"]) != count:
            raise RuntimeError("The existing trajectory cache has incompatible metadata")
        return payload["trajectories"]
    initial, _ = load_paper_fields(data_path, spec.resolution)
    if count > initial.shape[0]:
        raise ValueError("Requested trajectories exceed the paper dataset")
    solver = BurgersETDRK4(spec, device)
    chunks: list[torch.Tensor] = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        chunks.append(solver.solve(torch.from_numpy(initial[start:stop])).cpu())
        print(f"Generated trajectories {stop}/{count}")
    trajectories = torch.cat(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"spec": spec.__dict__, "count": count, "trajectories": trajectories},
        cache_path,
    )
    return trajectories
