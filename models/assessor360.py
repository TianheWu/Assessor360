import torch
import torch.nn as nn
import timm

from torch import nn
from einops import rearrange


def creat_model(config, pretrained=True):
    model_oiqa = Assessor360(num_layers=config.num_layers, embed_dim=config.embed_dim,
        dab_layers=config.dab_layers)
    if pretrained:
        model_oiqa.load_state_dict(torch.load(config.model_weight_path), strict=True)
    return model_oiqa


class ChannelAttn(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.act_layer = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _x = x
        x = self.gap(x)
        x = self.conv2(self.act_layer(self.conv1(x)))
        x = self.sigmoid(x)
        x = _x * x
        return x


class DAB(nn.Module):
    def __init__(self, embed_dim, dab_layers):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        channel_attn = []
        for i in range(dab_layers):
            channel_attn += [ChannelAttn(embed_dim=embed_dim)]
        self.channel_attn = nn.Sequential(*channel_attn)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attn(x)
        x = self.gap(x)
        return x


class TMM(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=num_layers, batch_first=True)
        self.fc_quality_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        B, M, N, C = x.shape
        seqs = torch.tensor([]).cpu()
        for i in range(B):
            seq = x[i]
            seq, _h = self.gru(seq)
            seqs = torch.cat((seqs, seq.unsqueeze(0).cpu()), dim=0)
        seqs = seqs.cuda() # batch, num_sequence, num_viewport, dim
        mem_quality = seqs[:, :, -1, :]
        mem_quality = self.fc_quality_score(mem_quality).flatten(1)
        return mem_quality


class Assessor360(nn.Module):
    def __init__(self, num_layers=6, embed_dim=128, dab_layers=6):
        super().__init__()
        self.act_layer = nn.GELU()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

        self.conv1_1 = nn.Conv2d(256, embed_dim, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(512, embed_dim, 1, 1, 0)
        self.conv3_1 = nn.Conv2d(1024, embed_dim, 1, 1, 0)
        self.conv4_1 = nn.Conv2d(1024, embed_dim, 1, 1, 0)

        self.dab1 = DAB(embed_dim=embed_dim, dab_layers=dab_layers)
        self.dab2 = DAB(embed_dim=embed_dim, dab_layers=dab_layers)
        
        self.conv_dis = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.conv_scene = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.conv_feat = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.tmm = TMM(embed_dim=embed_dim, num_layers=num_layers)

    def forward(self, x):
        B, M, N, C, H, W = x.shape

        # sequence features
        quality_feats = torch.tensor([]).cpu()
        for i in range(B):
            feats = torch.tensor([]).cpu()
            for j in range(M):
                x3, x0, x1, x2 = self.backbone(x[i][j])

                x0 = rearrange(x0, 'b (h w) c -> b c h w', h=28, w=28)
                x1 = rearrange(x1, 'b (h w) c -> b c h w', h=14, w=14)
                x2 = rearrange(x2, 'b (h w) c -> b c h w', h=7, w=7)
                x3 = rearrange(x3, 'b (h w) c -> b c h w', h=7, w=7)

                x0 = self.dab1(self.conv1_1(x0))
                x1 = self.dab2(self.conv2_1(x1))
                x2 = self.gap(self.conv3_1(x2))
                x3 = self.gap(self.conv4_1(x3))

                x_dis = self.conv_dis(torch.cat((x0, x1), dim=1))
                x_scene = self.conv_scene(torch.cat((x2, x3), dim=1))
                x_feat = self.conv_feat(torch.cat((x_dis, x_scene), dim=1))
                x_feat = x_feat.flatten(1).unsqueeze(0).cpu()

                # concat
                feats = torch.cat((feats, x_feat), dim=0)
            
            quality_feats = torch.cat((quality_feats, feats.unsqueeze(0)), dim=0)
        quality_feats = quality_feats.cuda() # B, M, N, C

        x = self.tmm(quality_feats) # B, M, C
        x = torch.mean(x, dim=1)
        return x