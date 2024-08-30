import math
import warnings

import torch.nn as nn

from models.base_model import BaseModel
from models.tool_models.NormalizedCutter import NCutterModule
from myUtils.config import Config
from myUtils.others import *
from losses.NCut_loss import NCutterLoss


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MultiViewSelfAttention(nn.Module):
    def __init__(self, dim, block_num=12, num_heads=6, qkv_bias=True, block_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_num = block_num
        self.dim = dim

        if block_size is None:
            block_size = [64, 64]

        self.pos_token_size = block_size

        self.pos_token = nn.Parameter(torch.zeros(1, 1, self.block_num, self.dim))
        trunc_normal_(self.pos_token, std=.02)

        self.block_proj_q = nn.Linear(dim * self.block_num, dim, bias=qkv_bias)
        self.block_proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.block_proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.ly_norm = nn.LayerNorm([self.dim])

    def forward(self, x, p_size):
        b, t_1, k, d = x.shape
        t = t_1 - 1

        # x_cls = x[:, 0, :, :]  # [b, k, d]
        x = x[:, 1:, :, :]  # [b, t, k, d]

        pos_token = self.pos_token

        x += pos_token

        x_q = self.block_proj_q(x.reshape(b, t, k * d)).reshape(b, t, self.num_heads, self.head_dim)
        x_k = self.block_proj_k(x).reshape(b, t, k, self.num_heads, self.head_dim)
        x_v = self.block_proj_v(x).reshape(b, t, k, self.num_heads, self.head_dim)

        cls_a = torch.einsum("btkhd, bthd -> btkh", x_k, x_q).mul(self.scale).softmax(dim=-2)

        x_k_a = x_v.mul(cls_a.reshape(b, t, k, self.num_heads, 1)).sum(dim=-3).reshape(b, t, self.dim)

        x_k_a = self.proj(x_k_a)

        cls_a_merged = cls_a.mean(dim=-1)
        if self.num_heads != 1:
            cls_a_merged = F.softmax(cls_a_merged, dim=-1)

        x_k_a = self.ly_norm(x_k_a)

        return x_k_a, cls_a_merged


class AMNCutter(BaseModel):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.method_repo = self.config.method_config['repo']
        self.method_name = self.config.method_config['name']

        self.method_config = self.config.method_config

        if "vit" in self.method_name:
            self.model = torch.hub.load(self.method_repo, self.method_name)

            self.patch_size = self.model.patch_embed.patch_size
            self.num_heads = self.model.blocks[0].attn.num_heads

            self.n_dim_feat = self.model.blocks[0].attn.qkv.out_features // 3

            self.block_num = len(self.model._modules["blocks"])

            self.extractor = self.extractor_dino
        else:
            raise ValueError(f"Unsupported name {self.method_name}")

        self.attention_ncut_mark = self.config.method_config.get("multi_view_attn")

        if self.attention_ncut_mark:
            self.attention_merge_module = MultiViewSelfAttention(dim=self.n_dim_feat, block_num=self.block_num, num_heads=6, qkv_bias=True)
            self.get_affinity = self.__get_affinity_attn__
        else:
            self.attention_merge_module = nn.Identity()
            self.get_affinity = self.__get_affinity_normal__

        self.loss_detail_key_list = ["Loss_Sum", "Loss_NCut"]

        self.normalized_cutter = NCutterModule(self.config, self.n_dim_feat)
        self.cutter_dim = self.normalized_cutter.cut_dim

        self.frozen_module_list = nn.ModuleList([self.model])
        self.trained_module_list = nn.ModuleList([self.normalized_cutter, self.attention_merge_module])

        self.loss_func_ncut = NCutterLoss()

        self.aff_mat_op_name = self.config.method_config.get("aff_mat_op")

        if self.aff_mat_op_name == "none":
            self.aff_mat_op = lambda x: x
        elif self.aff_mat_op_name == "set0":
            self.aff_mat_op = self.aff_mat_set0
        elif self.aff_mat_op_name == "add1":
            self.aff_mat_op = self.aff_mat_add1
        elif self.aff_mat_op_name == "noise":
            self.aff_mat_op = self.aff_mat_noise
        else:
            raise ValueError(f"Unsupported cut method {self.aff_mat_op_name}")

    def extractor_dino(self, x):
        p = self.patch_size
        b, c, h, w = x.shape
        w_p_size, h_p_size = torch.div(w, p, rounding_mode="trunc").item(), torch.div(h, p,
                                                                                      rounding_mode="trunc").item()
        w_size, h_size = w_p_size * p, h_p_size * p

        p_size = (h_p_size, w_p_size)

        t = w_p_size * h_p_size

        x_resized = F.interpolate(x, size=[h_size, w_size], mode="bilinear")

        block_list = self.model.get_intermediate_layers(x_resized, n=self.block_num)

        if self.attention_ncut_mark:
            block_list = [block_i.unsqueeze(2) for block_i in block_list]
            block_cat = torch.cat(block_list, dim=2)
            feats, attn = self.attention_merge_module(block_cat, p_size)
        else:
            feats = block_list[-1][:, 1:, :]
            attn = None
            block_cat = None

        return feats, p_size, block_cat, attn

    def forward(self, x, clear_hook=False):
        x = x.to(self.config.device)
        b, c, h, w = x.shape

        feature_map, p_size, block_cat, attn = self.extractor(x)

        h_p_size, w_p_size = p_size
        t = h_p_size * w_p_size

        cutting, seg_feat = self.normalized_cutter(feature_map.reshape(b, h_p_size, w_p_size, -1))

        cutting = cutting.reshape(b, h_p_size, w_p_size, -1)

        return cutting, feature_map, seg_feat, block_cat, attn

    @staticmethod
    def aff_mat_noise(affinity_matrix: torch.Tensor):
        affinity_matrix = affinity_matrix * 0 + torch.randn(size=affinity_matrix.shape, device=affinity_matrix.device)
        affinity_matrix = affinity_matrix.add(affinity_matrix.permute(0, 2, 1)).div(2)
        return affinity_matrix

    @staticmethod
    def aff_mat_add1(affinity_matrix: torch.Tensor):
        affinity_matrix = affinity_matrix.add(1).divide(2)
        return affinity_matrix

    @staticmethod
    def aff_mat_set0(affinity_matrix):
        affinity_matrix[affinity_matrix < 0] = 0
        return affinity_matrix

    def __get_affinity_normal__(self, feats, hwt, block_list, attn):
        h_p, w_p, t = hwt

        feats = feats.detach()

        feats_normed = F.normalize(feats, dim=-1)
        affinity_matrix = torch.bmm(feats_normed, feats_normed.permute(0, 2, 1))

        affinity_matrix = self.aff_tanh_opt(affinity_matrix, h_p, w_p)

        affinity_matrix = self.aff_mat_op(affinity_matrix)

        # for i in range(b):
        #     affinity_matrix[i].fill_diagonal_(0)

        affinity_matrix = affinity_matrix.triu(1) + affinity_matrix.tril(-1)

        return affinity_matrix, affinity_matrix, affinity_matrix

    def __get_affinity_attn__(self, feats, hwt, block_cat, attn: torch.Tensor):
        h_p, w_p, t = hwt
        b, t_1, k, d = block_cat.shape

        block_cat = block_cat.detach()

        attn = attn.detach()

        attn = attn.expand(size=[b, t, k])

        attn_sqrt = attn.sqrt()
        attn_matrix = torch.einsum("btik, biyk -> btyk", attn_sqrt.unsqueeze(2), attn_sqrt.unsqueeze(1))
        attn_matrix = attn_matrix.softmax(dim=-1)

        block_cat = F.normalize(block_cat[:, 1:, :, :], dim=-1)

        affinity_matrix_blocks = torch.einsum("btkd, bykd -> btyk", block_cat, block_cat)

        affinity_matrix = affinity_matrix_blocks.mul(attn_matrix)
        affinity_matrix = affinity_matrix.sum(dim=-1)

        affinity_matrix = self.aff_mat_op(affinity_matrix)
        affinity_matrix_blocks = self.aff_mat_op(affinity_matrix_blocks)

        affinity_matrix = affinity_matrix.triu(-2) + affinity_matrix.tril(-1)
        affinity_matrix_blocks = affinity_matrix_blocks.triu(1) + affinity_matrix_blocks.tril(2)

        return affinity_matrix, affinity_matrix_blocks, attn_matrix

    def get_loss(self, return_dict):
        loss_dict = {}

        x = return_dict[0]['x'].to(self.config.device)  # [b, c, h, w]
        b, c, h, w = x.shape

        c, f, s, block_list, attn = self.forward(x, clear_hook=True)

        h_p, w_p = c.shape[1:3]
        p_size = (h_p, w_p)
        t = h_p * w_p

        aff, _, _ = self.get_affinity(f, (h_p, w_p, t), block_list, attn)

        c = c.reshape(b, t, -1)

        loss_ncut = self.loss_func_ncut(aff, c)
        loss_dict["Loss_NCut"] = loss_ncut

        loss_sum = loss_ncut + 0.0

        loss_dict["Loss_Sum"] = loss_sum

        return loss_dict

    def get_seg(self, return_dict, vis=False):
        x = return_dict[0]['x'].to(self.config.device)

        b, c, h, w = x.shape

        cutting, feature_map, seg_feat, block_cat, attn = self.forward(x, clear_hook=True)

        h_p, w_p = cutting.shape[1:3]
        p_size = (h_p, w_p)
        t = h_p * w_p

        cutting = cutting.permute(0, 3, 1, 2)

        cutting = F.interpolate(cutting, (h, w), mode="bilinear")

        if vis:
            # It's a bit slow. if you are evaluating FPS, better no vis
            aff_fuse, aff_cat, attn_matrix = self.get_affinity(feature_map, (h_p, w_p, t), block_cat, attn)
            aff = torch.cat([aff_fuse.unsqueeze(-1), aff_cat], dim=-1).permute(0, 3, 1, 2)
            aff = F.interpolate(aff, (512, 512)).permute(0, 2, 3, 1)
            attn_matrix = F.interpolate(attn_matrix.permute(0, 3, 1, 2), (512, 512)).permute(0, 2, 3, 1)

            feature_map_vis = feature_map.reshape(b, h_p, w_p, -1)
            block_cat_vis = block_cat[:, 1:, :, :].reshape(b, h_p, w_p, self.block_num, -1).transpose(-1, -2)

            pro_pre = F.softmax(cutting, dim=1).permute(0, 2, 3, 1)

            return_dict[0]['feats'] = feature_map_vis
            return_dict[0]['affinity_matrix_attn'] = attn_matrix
            return_dict[0]['pro_pre'] = pro_pre
            return_dict[0]['feats_multi_block'] = block_cat_vis
            return_dict[0]['affinity_matrix'] = aff

        pre_seg = cutting.argmax(dim=1, keepdim=True).permute(0, 2, 3, 1).long()

        if attn is not None and block_cat is not None:

            b, t_1, k, d = block_cat.shape
            b_a, t_a, k_a = attn.shape
            attn_vis = attn.detach().expand([b, t, k])

            if t_a == 1:
                attn_vis = normalize_01_tensor(attn_vis, axis=2)
            else:
                attn_vis = normalize_01_tensor(attn_vis, axis=1)

            attn_vis = attn_vis.reshape(b, h_p, w_p, k).permute(0, 3, 1, 2)

            return_dict[0]["attn_vis"] = attn_vis

        return_dict[0]['pre_seg'] = pre_seg
        return_dict[0]['pre_seg_raw'] = torch.clone(return_dict[0]['pre_seg'].detach())

        return return_dict

    def training_vis(self, return_dict, img_idx):

        x0 = return_dict[0]['x'].to(self.config.device).unsqueeze(0)  # [b, c, h, w]
        b, c, h, w = x0.shape

        cs0, f0, s0, block_cat0, attn0 = self.forward(x0[0:1], clear_hook=True)

        h_p, w_p, d = cs0.shape[1:]
        t = h_p * w_p

        cs0 = cs0.reshape(1, t, -1)

        cs0 = cs0.transpose(1, 2).reshape(1, -1, h_p, w_p)[0]

        cs0 = F.softmax(cs0, dim=0)
        cs0 = normalize_01_tensor(cs0, axis=[1, 2])

        cs0 = TF.resize(cs0, size=[512, 512], interpolation=TF.InterpolationMode.NEAREST_EXACT)

        h_p, w_p = cs0.shape[1:]

        image_r = int(np.ceil(d ** 0.5))

        remain_black_num = image_r**2 - d

        black_image = torch.zeros(size=(remain_black_num, h_p, w_p), dtype=cs0.dtype, device=cs0.device)

        vis_img_flatten = torch.cat([cs0, black_image], dim=0)

        vis_img_list1 = []
        for i in range(image_r):
            vis_img_list2 = []
            for j in range(image_r):
                ij = i * image_r + j
                vis_img_list2.append(vis_img_flatten[ij])
            vis_img_list2 = torch.cat(vis_img_list2, dim=1)
            vis_img_list1.append(vis_img_list2)
        vis_img_list1 = torch.cat(vis_img_list1, dim=0)

        vis_img = (vis_img_list1.cpu().numpy() * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_VIRIDIS)

        # cv2.imshow("training_vis", vis_img)
        save_path = f"training_vis_cache/{self.config.config_file_path_name}/{self.config.config_file_name}"
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(f"{save_path}/{img_idx}.png", vis_img)

