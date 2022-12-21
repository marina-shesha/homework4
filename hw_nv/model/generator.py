import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm


class GenBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for i in range(len(dilation)):
            padding = int((kernel_size * dilation[i] - dilation[i]) / 2)
            self.convs1.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=padding
                ))
            ))
        self.convs2 = nn.ModuleList()
        for i in range(len(dilation)):
            padding = int((kernel_size - 1) / 2)
            self.convs2.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=padding
                ))
            ))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, input):
        for c1, c2 in zip(self.convs1, self.convs2):
            out = c1(input)
            out = c2(out)
            input = input + out
        return input


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.channels_up
        self.conv1 = weight_norm(nn.Conv1d(80, ch, 7, 1, padding=3))
        self.convT = nn.ModuleList()
        self.res_block = nn.ModuleList()
        self.l = len(config.blok_kernel_size)
        for ks, st in zip(config.kernel_size_convT, config.stride_convT):
            self.convT.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.ConvTranspose1d(ch, ch // 2, ks, st, padding=(ks-st)//2))
                )
            )
            for ks, d in zip(config.blok_kernel_size, config.blok_dilation):
                self.res_block.append(GenBlock(ch//2, ks, d))
            ch = ch // 2

        self.conv2 = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.convT.apply(init_weights)

    def forward(self, input):
        out = self.conv1(input)
        i = 0
        for layer in self.convT:
            out = layer(out)
            block_out = None
            for j in range(self.l):
                if block_out is None:
                    block_out = self.res_block[i*self.l + j](out)
                else:
                    block_out = block_out + self.res_block[i*self.l + j](out)
            out = block_out / self.l
            i += 1
        out = F.leaky_relu(out, 0.1)
        out = self.conv2(out)
        out = torch.tanh(out)

        return out


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
