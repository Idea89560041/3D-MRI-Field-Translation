import torch
import torch.nn.functional as F


def hinge_discriminator_loss(real_logits, fake_logits):
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def hinge_generator_loss(fake_logits):
    return -fake_logits.mean()


def gradient_l1(a, b):
    loss = 0.0
    for dim in (2, 3, 4):
        da = a.diff(dim=dim)
        db = b.diff(dim=dim)
        loss = loss + F.l1_loss(da, db)
    return loss / 3.0

