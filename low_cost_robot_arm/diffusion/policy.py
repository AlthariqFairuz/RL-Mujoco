"""State-based Diffusion Policy (Chi et al., RSS 2023), written out in plain PyTorch.

The policy sees the last `OBS_HORIZON` observations and generates a chunk of `PRED_HORIZON`
future joint targets by denoising Gaussian noise. The first `ACTION_HORIZON` targets are
executed, then it plans again (receding horizon).

    noise prediction network   1D temporal U-Net over the target sequence, conditioned on the
                               observations and the diffusion step through FiLM
    training                   DDPM: add noise of a random step k to an expert chunk, predict
                               the noise (MSE)
    inference                  DDIM: 100 training steps, sampled in `INFER_STEPS` steps
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
TRAIN_STEPS, INFER_STEPS = 100, 10


# ------------------------------------------------------------------ noise schedule (DDPM/DDIM)
def cosine_alphas_cumprod(steps: int) -> torch.Tensor:
    """Squared-cosine schedule (Nichol & Dhariwal 2021), betas capped at 0.999."""
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


# ------------------------------------------------------------------ network
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10_000) * torch.arange(half, device=k.device) / (half - 1))
        x = k.float()[:, None] * freq[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class Conv1dBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel: int = 5, groups: int = 8):
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.GroupNorm(groups, c_out),
            nn.Mish(),
        )


class CondResBlock(nn.Module):
    """Two conv blocks; the condition scales and shifts the features in between (FiLM)."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.block1, self.block2 = Conv1dBlock(c_in, c_out), Conv1dBlock(c_out, c_out)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * c_out))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(cond)[..., None].chunk(2, dim=1)
        h = self.block1(x) * scale + bias
        return self.block2(h) + self.skip(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, dims=(128, 256, 512), emb_dim: int = 128):
        super().__init__()
        self.k_emb = nn.Sequential(
            SinusoidalPosEmb(emb_dim), nn.Linear(emb_dim, 4 * emb_dim), nn.Mish(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
        cond_dim += emb_dim
        chans = [action_dim, *dims]
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(zip(chans[:-1], chans[1:])):
            last = i == len(dims) - 1
            self.down.append(nn.ModuleList([
                CondResBlock(a, b, cond_dim), CondResBlock(b, b, cond_dim),
                nn.Identity() if last else nn.Conv1d(b, b, 3, stride=2, padding=1),
            ]))
        self.mid = nn.ModuleList([CondResBlock(dims[-1], dims[-1], cond_dim) for _ in range(2)])
        self.up = nn.ModuleList()
        for a, b in zip(reversed(dims[1:]), reversed(dims[:-1])):
            self.up.append(nn.ModuleList([
                CondResBlock(2 * a, b, cond_dim), CondResBlock(b, b, cond_dim),
                nn.ConvTranspose1d(b, b, 4, stride=2, padding=1),
            ]))
        self.final = nn.Sequential(Conv1dBlock(dims[0], dims[0]), nn.Conv1d(dims[0], action_dim, 1))

    def forward(self, x: torch.Tensor, k: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """x (B, T, action_dim) noisy chunk, k (B,) diffusion step, obs_cond (B, cond_dim)."""
        cond = torch.cat([self.k_emb(k), obs_cond], dim=-1)
        x = x.transpose(1, 2)
        skips = []
        for res1, res2, downsample in self.down:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for res in self.mid:
            x = res(x, cond)
        for res1, res2, upsample in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = upsample(res2(res1(x, cond), cond))
        return self.final(x).transpose(1, 2)


# ------------------------------------------------------------------ policy
class DiffusionPolicy(nn.Module):
    """Noise-prediction network + observation normaliser + DDPM loss / DDIM sampler.

    Targets are already in [-1, 1]; observations are min-max scaled to [-1, 1] with the
    dataset statistics, which are stored as buffers so they are saved with the weights.
    """

    def __init__(self, obs_dim: int, action_dim: int, obs_min=None, obs_max=None):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.net = ConditionalUnet1D(action_dim, OBS_HORIZON * obs_dim)
        self.register_buffer("obs_min", torch.as_tensor(obs_min if obs_min is not None
                                                        else -np.ones(obs_dim)).float())
        self.register_buffer("obs_max", torch.as_tensor(obs_max if obs_max is not None
                                                        else np.ones(obs_dim)).float())
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(TRAIN_STEPS))

    def _cond(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, OBS_HORIZON, obs_dim) -> normalised and flattened (B, OBS_HORIZON * obs_dim)."""
        span = (self.obs_max - self.obs_min).clamp(min=1e-6)
        return (2 * (obs - self.obs_min) / span - 1).flatten(1)

    def loss(self, obs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b = targets.shape[0]
        k = torch.randint(0, TRAIN_STEPS, (b,), device=targets.device)
        noise = torch.randn_like(targets)
        a = self.alphas_cumprod[k][:, None, None]
        noisy = a.sqrt() * targets + (1 - a).sqrt() * noise
        return F.mse_loss(self.net(noisy, k, self._cond(obs)), noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, steps: int = INFER_STEPS) -> torch.Tensor:
        """Deterministic DDIM (eta = 0): (B, OBS_HORIZON, obs_dim) -> (B, PRED_HORIZON, 6)."""
        cond = self._cond(obs)
        x = torch.randn(obs.shape[0], PRED_HORIZON, self.action_dim, device=obs.device)
        ks = torch.linspace(TRAIN_STEPS - 1, 0, steps, device=obs.device).round().long()
        for i, k in enumerate(ks):
            a = self.alphas_cumprod[k]
            a_prev = self.alphas_cumprod[ks[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            eps = self.net(x, k.expand(x.shape[0]), cond)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # consistent with the clipped x0
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
