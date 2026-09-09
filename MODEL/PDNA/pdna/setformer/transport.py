from __future__ import annotations

import torch
from torch import Tensor, nn


def squared_distance(x: Tensor, y: Tensor) -> Tensor:
    return (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)


@torch.no_grad()
def kmeans_centroids(samples: Tensor, clusters: int, iterations: int = 50) -> Tensor:
    """K-means initialization required by Section 3.4 for the reference set."""
    if samples.ndim != 2 or samples.shape[0] < clusters:
        raise ValueError("K-means needs [samples, features] with samples >= clusters")
    order = torch.randperm(samples.shape[0], device=samples.device)
    centroids = samples[order[:clusters]].clone()
    for _ in range(iterations):
        assignment = torch.cdist(samples, centroids).argmin(1)
        updated = torch.stack([
            samples[assignment == index].mean(0)
            if (assignment == index).any() else centroids[index]
            for index in range(clusters)
        ])
        if torch.allclose(updated, centroids, atol=1e-5, rtol=1e-4):
            break
        centroids = updated
    return centroids


class NystromRBFKernel(nn.Module):
    """Low-rank RBF feature map used before optimal-transport cost construction."""

    def __init__(self, dimension: int, landmarks: int = 32, gamma: float | None = None):
        super().__init__()
        self.gamma = 1.0 / dimension if gamma is None else gamma
        self.landmarks = nn.Parameter(torch.randn(landmarks, dimension) / dimension**0.5)

    def rbf(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.exp(-self.gamma * squared_distance(x, y))

    def feature_map(self, x: Tensor) -> Tensor:
        k_xm = self.rbf(x, self.landmarks)
        k_mm = self.rbf(self.landmarks, self.landmarks)
        eigenvalues, eigenvectors = torch.linalg.eigh(k_mm.float())
        inverse_sqrt = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(1e-6).rsqrt()) @ eigenvectors.T
        return k_xm @ inverse_sqrt.to(k_xm.dtype)


class SinkhornSolver(nn.Module):
    def __init__(self, epsilon: float = 0.1, iterations: int = 50):
        super().__init__()
        self.epsilon, self.iterations = epsilon, iterations

    def forward(self, cost: Tensor, source: Tensor | None = None,
                target: Tensor | None = None) -> Tensor:
        batch, n, m = cost.shape
        dtype, device = cost.dtype, cost.device
        source = (torch.full((batch, n), 1.0 / n, dtype=dtype, device=device)
                  if source is None else source)
        target = (torch.full((batch, m), 1.0 / m, dtype=dtype, device=device)
                  if target is None else target)
        log_k = -cost / self.epsilon
        log_u = torch.zeros_like(source)
        log_v = torch.zeros_like(target)
        for _ in range(self.iterations):
            log_u = source.clamp_min(1e-12).log() - torch.logsumexp(
                log_k + log_v.unsqueeze(1), dim=2
            )
            log_v = target.clamp_min(1e-12).log() - torch.logsumexp(
                log_k + log_u.unsqueeze(2), dim=1
            )
        return torch.exp(log_k + log_u.unsqueeze(2) + log_v.unsqueeze(1))


class SelfOptimalTransport(nn.Module):
    def __init__(self, dimension: int = 128, reference_points: int = 64,
                 landmarks: int = 32, rbf_gamma: float | None = None,
                 epsilon: float = 0.1, iterations: int = 50):
        super().__init__()
        self.references = nn.Parameter(torch.randn(reference_points, dimension) / dimension**0.5)
        self.kernel = NystromRBFKernel(dimension, landmarks, rbf_gamma)
        self.sinkhorn = SinkhornSolver(epsilon, iterations)

    @torch.no_grad()
    def initialize_references(self, centroids: Tensor) -> None:
        if centroids.shape != self.references.shape:
            raise ValueError(f"Expected centroids {tuple(self.references.shape)}, got {tuple(centroids.shape)}")
        self.references.copy_(centroids)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        references = self.references.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        phi_x = self.kernel.feature_map(inputs)
        phi_r = self.kernel.feature_map(references)
        cost = squared_distance(phi_x, phi_r)
        plan = self.sinkhorn(cost)
        # Conditional barycentric projection prevents uniform mass from shrinking magnitudes.
        aligned = plan @ references / plan.sum(-1, keepdim=True).clamp_min(1e-12)
        return aligned, plan, cost
