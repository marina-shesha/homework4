import torch
import torch.nn.functional as F


def feat_loss(true_feats, fake_feats):
    loss = 0
    for true_feat, fake_feat in zip(true_feats, fake_feats):
        for true, false in zip(true_feat, fake_feat):
            loss += F.l1_loss(true, false)
    loss *= 2
    return loss


def discriminator_loss(true_out, fake_out):
    loss = 0
    for true, fake in zip(true_out, fake_out):
        loss += torch.mean((1-true)**2) + torch.mean(fake**2)
    return loss


def generator_loss(out):
    loss = 0
    for o in out:
        loss += torch.mean((1-o)**2)
    return loss
