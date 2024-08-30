from scipy.optimize import linear_sum_assignment
from dataloader import MyDataset
from myUtils.config import Config
from myUtils.model_config_list import SUP_METHOD_LIST
from myUtils.others import *


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


LABEL2RGB_COLOR_MAP = color_map(normalized=True).tolist()


class SegTool:
    def __init__(self, config: Config, dataset: MyDataset):
        self.config = config
        self.dataset = dataset
        self.dataset_config = self.dataset.dataset_config

        self.cluster_k = self.config.method_config.get("cluster_k")
        self.class_n = self.dataset_config.class_n

        self.pre_real_array = None
        self.pre_real_count = None
        self.pre_real_match = None

        if self.cluster_k is not None and self.cluster_k > 0:
            self.auto_cluster_k = False
            self.pre_real_array = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_count = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_match = torch.zeros(size=[self.cluster_k, 2], dtype=torch.int64, device=self.config.device)
        else:
            self.auto_cluster_k = True

        self.pre_real_array_dict = {}
        self.pre_real_count_dict = {}

    def _update_pre_real_array(self, pre_seg: torch.Tensor, gt_seg: torch.Tensor):
        pre = pre_seg.flatten().long().to(self.config.device)
        gt = gt_seg.flatten().long().to(self.config.device)

        for cluster_i in range(self.cluster_k):
            for class_i in range(self.class_n):
                pre_mask = pre == cluster_i
                gt_mask = gt == class_i
                inter_mask = torch.logical_and(pre_mask, gt_mask)
                uni_mask = torch.logical_or(pre_mask, gt_mask)
                self.pre_real_array[cluster_i, class_i] += inter_mask.sum()
                self.pre_real_count[cluster_i, class_i] += uni_mask.sum()

    def get_match(self):
        """
        Majority Voting Match
        :return:
        """
        pre_real_iou = self.pre_real_array / (self.pre_real_count + 1e-10)
        if self.pre_real_array.shape[0] == self.pre_real_array.shape[1]:
            _, assignments = linear_sum_assignment(1 - pre_real_iou.cpu().numpy())
        else:
            assignments = torch.argmax(pre_real_iou, dim=1)
        for i in range(self.cluster_k):
            self.pre_real_match[i, 0] = i
            self.pre_real_match[i, 1] = assignments[i]

    def label_free_postprocessing(self, return_dict, idx, match=True, single_match=False):
        pre_seg = return_dict["pre_seg"][idx]
        gt_seg = return_dict["gt"][idx].to(self.config.device)

        if match:

            if single_match:
                self.hungarian_match_init(k=pre_seg.max().item())
                self._update_pre_real_array(pre_seg, gt_seg)
                self.get_match()

            good_pre_seg = torch.zeros_like(pre_seg)

            for i, j in self.pre_real_match:
                good_pre_seg[pre_seg == i] = j
        else:
            good_pre_seg = pre_seg

        return good_pre_seg

    def hungarian_match_init(self, k=256):
        if not self.auto_cluster_k:
            # pass
            self.pre_real_array *= 0
            self.pre_real_count *= 0
        else:
            self.cluster_k = k
            self.pre_real_array = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_count = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_match = torch.zeros(size=[self.cluster_k, 2], dtype=torch.int64, device=self.config.device)

    def hungarian_match_record(self, batch_dict):
        pre_seg = batch_dict["pre_seg"].to(self.config.device)
        gt_seg = batch_dict["gt"].to(self.config.device)
        b = len(pre_seg)
        for idx in range(b):
            self._update_pre_real_array(pre_seg[idx], gt_seg[idx])

    def hungarian_get_match(self):
        self.get_match()

    def seg_postprocessing(self, return_dict_i, match=True, single_match=False):
        if self.config.method_name in SUP_METHOD_LIST:
            pass
        else:
            for i in range(len(return_dict_i["pre_seg"])):
                return_dict_i["pre_seg"][i] = self.label_free_postprocessing(return_dict_i, i, match=match, single_match=single_match)
