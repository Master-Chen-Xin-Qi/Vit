# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :Vit
# @File     :models
# @Date     :2022/1/18 17:03
# @Author   :Xinqi Chen
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2)  -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
            # nn.Linear(patch_size*patch_size*in_channels, emb_size)
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.embedding(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions
        return x


class MultiAttn(nn.Module):
    def __init__(self, emb_size=512, head=8, dropout=0.8):
        super(MultiAttn, self).__init__()
        self.emb_size = emb_size
        self.head = head
        self.q = nn.Linear(self.emb_size, self.emb_size)
        self.k = nn.Linear(self.emb_size, self.emb_size)
        self.v = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, x, mask=None):
        q = rearrange(self.q(x), 'b q (n d) -> b n q d', n=self.head)
        k = rearrange(self.k(x), 'b q (n d) -> b n q d', n=self.head)
        v = rearrange(self.v(x), 'b q (n d) -> b n q d', n=self.head)
        energy = torch.einsum('bnqd, bnkd -> bnqk', q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.dropout(att)
        out = torch.einsum('bnqk, bnvd -> bnqd', att, v)
        out = rearrange(out, "b n q d -> b q (n d)")
        out = self.projection(out)
        return out


class ResAdd(nn.Module):
    def __init__(self, fn):
        super(ResAdd, self).__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x


class Feedforward(nn.Module):
    def __init__(self, embed_size=768, expasion=4, dropout=0):
        super(Feedforward, self).__init__()
        self.l1 = nn.Linear(embed_size, embed_size*expasion)
        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(embed_size*expasion, embed_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(self.activ(x))
        x = self.l2(x)
        return x


class TransformerEnoderBlock(nn.Module):
    def __init__(self, embed_size=768, head=8, expansion=4, dropout1=0, dropout2=0, dropoutp=0):
        super(TransformerEnoderBlock, self).__init__()
        self.embed_size = embed_size
        self.head = head
        self.expansion = expansion
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.multiattn = MultiAttn(self.embed_size, self.head, self.dropout1)
        self.ff = Feedforward(self.embed_size, self.expansion, self.dropout2)
        self.ln1 = nn.LayerNorm(self.embed_size)
        self.ln2 = nn.LayerNorm(self.embed_size)
        self.res1 = ResAdd(nn.Sequential(
            self.ln1,
            self.multiattn,
            nn.Dropout(dropoutp)
        ))
        self.res2 = ResAdd(nn.Sequential(
            self.ln2,
            self.ff,
            nn.Dropout(dropoutp)
        ))

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n=4):
        super(TransformerEncoder, self).__init__()
        self.n = n
        self.model = nn.Sequential(*[TransformerEnoderBlock() for i in range(self.n)])

    def forward(self, x):
        return self.model(x)


class MLPClassifier(nn.Module):
    def __init__(self, embed_size=768, class_num=1000):
        super(MLPClassifier, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.avg = Reduce('b n e -> b e', reduction='mean')
        self.linear = nn.Linear(embed_size, class_num)

    def forward(self, x):
        x = self.norm(x)
        x = self.avg(x)
        x = self.linear(x)
        return x


class Vit(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, n=4, class_num=1000):
        super(Vit, self).__init__()
        self.layer = nn.Sequential(
            PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size),
            TransformerEncoder(n=n),
            MLPClassifier(embed_size=emb_size, class_num=class_num)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == '__main__':
    data = torch.randn((1, 3, 224, 224))
    data = Vit()(data)
    print(data.shape)
    summary(Vit(), (3, 224, 224), device='cpu')
