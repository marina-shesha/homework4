from .generator import Generator
from .discriminator import MultiPeriodDiscriminator
from .discriminator import MultiScaledDiscriminator
from typing import List
import torch.nn as nn
from ..loss.losses import feat_loss, generator_loss, discriminator_loss
from ..data.MelSpec import MelSpectrogram
import torch.nn.functional as F


class ModelConfig:
    kernel_size_convT: List[int] = [16, 16, 4, 4],
    stride_convT: List[int] = [8, 8, 2, 2],
    blok_kernel_size:  List[int] = [3, 7, 11],
    blok_dilation: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    channels_up: int = 128


class HiFiGAN(nn.Module):
    def __init__(self, config = ModelConfig()):
        super().__init__()

        self.g_model = Generator(config)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaledDiscriminator()

        self.melspec = MelSpectrogram()

    def forward(self, input):
        return self.g_model(input)

    def optimizer_step(self, batch, optimizer_g, optimizer_d):
        true_mel= batch['melspec']
        true_wav = batch['audio']

        fake_wav = self.g_model(true_mel)

        d_loss, mpd_loss, msd_loss = self.discriminator_step(optimizer_d, true_mel, true_wav, fake_wav)
        g_loss, mel_loss = self.generator_step(optimizer_g, true_mel, true_wav, fake_wav)

        return d_loss, mpd_loss, msd_loss, g_loss, mel_loss

    def discriminator_step(self, optimizer_d, true_melspec, true_wav, fake_wav):
        optimizer_d.zero_grad()

        true_outs, fake_outs, true_feat_maps, fake_feat_maps = self.mpd(true_wav, fake_wav.detach())
        mpd_loss = discriminator_loss(true_outs, fake_outs)

        true_outs, fake_outs, true_feat_maps, fake_feat_maps = self.msd(true_wav, fake_wav.detach())
        msd_loss = discriminator_loss(true_outs, fake_outs)

        loss = mpd_loss + msd_loss

        loss.backward()

        optimizer_d.step()

        return loss, mpd_loss, msd_loss

    def generator_step(self, optimizer_g, true_mel, true_wav, fake_wav):
        optimizer_g.zero_grad()

        fake_mel = self.melspec(fake_wav)

        mel_loss = F.l1_loss(true_mel, fake_mel) * 45

        true_outs, fake_outs, true_feat_maps, fake_feat_maps = self.mpd(true_wav, fake_wav)
        mpd_feat_loss = feat_loss(true_feat_maps, fake_feat_maps)
        mpd_gen_loss = generator_loss(fake_outs)

        true_outs, fake_outs, true_feat_maps, fake_feat_maps = self.msd(true_wav, fake_wav)
        msd_feat_loss = feat_loss(true_feat_maps, fake_feat_maps)
        msd_gen_loss = generator_loss(fake_outs)

        loss = mpd_feat_loss + mpd_gen_loss + msd_feat_loss + msd_gen_loss + mel_loss

        loss.backward()
        optimizer_g.step()

        return loss, mel_loss








