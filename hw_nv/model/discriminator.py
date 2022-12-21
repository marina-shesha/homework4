import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        ks = 5
        stride = 3
        self.p = period
        padding = padding = int((5 - 1) / 2)
        self.convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv2d(1, 32, (ks, 1), (stride, 1), padding=(padding, 0))),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(32, 128, (ks, 1), (stride, 1), padding=(padding, 0))),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(128, 512, (ks, 1), (stride, 1), padding=(padding, 0))),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(512, 1024, (ks, 1), (stride, 1), padding=(padding, 0))),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(1024, 1024, (ks, 1), 1, padding=(2, 0))),
                nn.LeakyReLU(0.1)
            )
        ])
        self.convs.append(weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))))

    def forward(self, input):
        batch, channel, time = input.shape
        out = input
        if time % self.p != 0:
            pad = self.p - time % self.p
            out = F.pad(out, (0, pad), "reflect")
        out = out.view(batch, channel, -1, self.p)

        feat = []
        for layer in self.convs:
            out = layer(out)
            feat.append(out)
        out = torch.flatten(out, start_dim=1)
        return out, feat


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_discrs = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11)
        ])

    def forward(self, true_input, fake_input):
        true_outs = []
        fake_outs = []
        true_feats = []
        fake_feats = []
        for layer in self.p_discrs:
            true_out, true_feat = layer(true_input)
            fake_out, fake_feat = layer(fake_input)
            true_outs.append(true_out)
            fake_outs.append(fake_out)
            true_feats.append(true_feat)
            fake_feats.append(fake_feat)
        return true_outs, fake_outs, true_feats, fake_feats


class ScaledDiscriminator(nn.Module):
    def __init__(self, norm=weight_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                nn.LeakyReLU(0.1)
            ),
        ])
        self.convs.append(weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)))

    def forward(self, input):
        feat = []
        out = input
        for layer in self.convs:
            out = layer(out)
            feat.append(out)
        out = torch.flatten(out, start_dim=1)
        return out, feat


class MultiScaledDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.s_discrs = nn.ModuleList([
            ScaledDiscriminator(spectral_norm),
            ScaledDiscriminator(),
            ScaledDiscriminator()
        ])
        self.polling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, true_input, fake_input):
        true_outs = []
        fake_outs = []
        true_feats = []
        fake_feats = []
        i = 0
        for layer in self.s_discrs:

            true_out, true_feat = layer(true_input)
            fake_out, fake_feat = layer(fake_input)
            true_outs.append(true_out)
            fake_outs.append(fake_out)
            true_feats.append(true_feat)
            fake_feats.append(fake_feat)
            if i < len(self.polling):
                true_input = self.polling[i](true_input)
                fake_input = self.polling[i](fake_input)
            i += 1

        return true_outs, fake_outs, true_feats, fake_feats


