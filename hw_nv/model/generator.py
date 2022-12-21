import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm


class GenBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        convs1 = []
        for i in range(len(dilation)):
            convs1.append(
                nn.LeakyReLU(0.1)
            )
            padding = int((kernel_size * dilation[i] - dilation[i])/2)
            convs1.append(
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[i],
                    padding=padding
                ))
            )
        convs2 = []
        for i in range(len(dilation)):
            convs2.append(
                nn.LeakyReLU(0.1)
            )
            padding = int((kernel_size * dilation[i] - 1) / 2)
            convs2.append(
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=padding
                ))
            )
        self.convs1 = nn.ModuleList(convs1)
        self.convs2 = nn.ModuleList(convs2)
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, input):
        out = self.convs1(input)
        out = self.convs2(out)
        out += input
        return out


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.channels_up
        self.conv1 = weight_norm(nn.Conv1d(80, ch, 7, 1, padding=3))
        self.convT = nn.ModuleList()
        self.res_block = nn.ModuleList()
        for ks, st in zip(config.kernel_size_convT, config.stride_convT):
            self.convT.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.ConvTranspose1d(ch, ch // 2, ks, st, padding=(ks-st)//2))
                )
            )
            for ks, d in zip(config.blok_kernel_size, config.blok_dilation):

                self.res_bloks.append(GenBlock(ch//2, ks, d))
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
            l = len(self.convT)
            for j in range(l):
                if block_out is None:
                    block_out = self.res_block[i*l + j](out)
                else:
                    block_out += self.res_block[i*l + j](out)
            out = block_out / l
            i += 1
        out = F.leaky_relu(out, 0.1)
        out = self.conv2(out)
        out = torch.tanh(out)

        return out


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
