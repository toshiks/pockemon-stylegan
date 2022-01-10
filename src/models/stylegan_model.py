from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import src.models.utils as utils
import src.models.adaptive_augment as aa
from .modules.stylegan import Discriminator, Generator


class StyleGanModel(pl.LightningModule):
    def __init__(
            self,
            image_size: int = 128,
            latent: int = 512,
            n_mlp: int = 8,

            g_reg_interval: int = 4,
            d_reg_interval: int = 16,

            lr: float = 0.002,
            batch: int = 16
    ):
        super().__init__()

        self.save_hyperparameters()

        self.generator = Generator(
            size=image_size,
            style_dim=latent,
            n_mlp=n_mlp
        )
        self.discriminator = Discriminator(
            size=image_size
        )
        self.generator_ema = Generator(
            size=image_size,
            style_dim=latent,
            n_mlp=n_mlp
        )

        self.generator_ema.eval()
        self.generator_ema.requires_grad_(False)
        utils.accumulate(self.generator_ema, self.generator, 0)

        self.ada_augment = aa.AdaptiveAugment(0.6, 500000, 8)
        self.mean_path_length = 0

    def train_discriminator(self, batch: Any):
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        noise = utils.mixing_noise(self.hparams.batch, self.hparams.latent, 0.9, batch.device)
        fake_img, _ = self.generator(noise)

        real_img_aug, _ = aa.augment(batch, 0)
        fake_img, _ = aa.augment(fake_img, 0)

        fake_pred = self.discriminator(fake_img)
        real_pred = self.discriminator(real_img_aug)

        loss = F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()

        ada_aug_p = self.ada_augment.tune(real_pred)
        if self.global_step % self.hparams.d_reg_interval == 0:
            batch.requires_grad = True

            real_img_aug, _ = aa.augment(batch, ada_aug_p)
            real_pred = self.discriminator(real_img_aug)
            r1_loss = utils.d_r1_loss(real_pred, batch)

            loss += 5 * r1_loss * self.hparams.d_reg_interval

        return loss

    def train_generator(self, batch: Any):
        utils.accumulate(self.generator_ema, self.generator, 0.997784)

        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        noise = utils.mixing_noise(self.hparams.batch, self.hparams.latent, 0.9, batch.device)
        fake_img, _ = self.generator(noise)

        fake_img, _ = aa.augment(fake_img, 0)
        fake_pred = self.discriminator(fake_img)

        loss = F.softplus(-fake_pred).mean()

        if self.global_step % self.hparams.g_reg_interval == 0:
            noise = utils.mixing_noise(self.hparams.batch, self.hparams.latent, 0.9, batch.device)
            fake_img, latents = self.generator(noise, return_latents=True)

            path_loss, self.mean_path_length, path_lengths = utils.g_path_regularize(
                fake_img, latents, self.mean_path_length
            )

            weighted_path_loss = 2 * self.hparams.g_reg_interval * path_loss
            loss += weighted_path_loss

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        if optimizer_idx == 0:
            g_loss = self.train_generator(batch)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
        else:
            d_loss = self.train_discriminator(batch)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        g_reg_ratio = self.hparams.g_reg_interval / (self.hparams.g_reg_interval + 1)
        d_reg_ratio = self.hparams.d_reg_interval / (self.hparams.d_reg_interval + 1)

        g_optim = optim.AdamW(
            params=self.generator.parameters(),
            lr=self.hparams.lr * g_reg_ratio,
            betas=(0, 0.99 ** g_reg_ratio)
        )

        d_optim = optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr * d_reg_ratio,
            betas=(0, 0.99 ** d_reg_ratio)
        )

        return [g_optim, d_optim], []
