import denseCRF
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from skimage.color import label2rgb

from models.base_model import BaseModel
from models.tool_models.ViT import VisionTransformer
from myUtils.config import Config
from myUtils.others import *
from losses.NCut_loss import NCutterLoss
# from ViT import VisionTransformer


class NCutterModule(nn.Module):
    def __init__(self, config: Config, in_dim):
        super().__init__()
        self.config = config

        self.in_dim = in_dim
        self.cut_dim = self.config.method_config["ncutter_dim"]
        self.out_class_num = self.config.method_config["cluster_k"]

        self.layer_num = self.config.method_config.get("layer_num")

        self.cut_layer_name = self.config.method_config.get("cut_layer")
        if self.cut_layer_name == "trans":
            self.cutter_layer = VisionTransformer(patch_size=1, in_chans=self.in_dim, embed_dim=self.cut_dim, num_heads=3, depth=self.layer_num, qkv_bias=True, pos_emb=False)
            self.forward = self.forward_trans__
        elif self.cut_layer_name == "conv":
            self.cutter_layer = nn.Sequential(
                nn.Conv2d(self.in_dim, self.cut_dim, 1, 1),
            )

            for _ in range(self.layer_num):
                self.cutter_layer.append(
                    nn.Conv2d(self.cut_dim, self.cut_dim, 3, 1, padding="same", padding_mode="replicate")
                )
                self.cutter_layer.append(
                    nn.ReLU(inplace=True),
                )

            self.forward = self.forward_conv__
        else:
            raise ValueError(f"Unsupported cut method {self.cut_layer_name}")

        self.seg_head_linear = torch.nn.Sequential(
            nn.Conv2d(self.cut_dim, self.cut_dim, 3, 1, padding="same"),
        )

        self.output_layer = torch.nn.Sequential(
            nn.Conv2d(self.cut_dim, self.out_class_num, 1, 1, padding="same"),
        )

    def forward_trans__(self, inputs):
        b, h, w, d = inputs.shape

        feats_cut = inputs
        feats_cut = feats_cut.permute(0, 3, 1, 2)

        feature_map = self.cutter_layer(feats_cut)

        feature_map = feature_map.permute(0, 2, 1).reshape(b, -1, h, w)

        seg_map = self.seg_head_linear(feature_map)  # + self.seg_head_non_linear(feature_map)

        cutting = self.output_layer(seg_map)

        cutting = cutting.reshape(b, -1, h * w).permute(0, 2, 1)

        return cutting, feature_map

    def forward_conv__(self, inputs):
        b, h, w, d = inputs.shape

        feats_cut = inputs
        feature_map = feats_cut.permute(0, 3, 1, 2)
        feature_map = self.cutter_layer(feature_map)

        seg_map = self.seg_head_linear(feature_map)  # + self.seg_head_non_linear(feature_map)

        cutting = self.output_layer(seg_map)

        cutting = cutting.reshape(b, -1, h * w).permute(0, 2, 1)

        return cutting, feature_map
