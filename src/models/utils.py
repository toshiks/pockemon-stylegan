import math
import random

import torch
import torch.nn as nn

from .modules.stylegan.op import conv2d_gradfix


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def make_noise(batch: int, latent_dim: int, n_noise: int, device: torch.device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch: int, latent_dim: int, prob: float, device: torch.device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    return [make_noise(batch, latent_dim, 1, device)]


def accumulate(model1: nn.Module, model2: nn.Module, decay: float = 0.999):
    """Supporting function for EMA.

    Args:
        model1: first model
        model2: second model
        decay: coefficient for ema

    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for key in par1.keys():
        par1[key].data.mul_(decay).add_(par2[key].data, alpha=1 - decay)
