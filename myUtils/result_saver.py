import sys

from dataloader import MyDataset
from myUtils.config import Config

import matplotlib.pyplot as plt

from myUtils.others import *
from skimage.color import label2rgb

sys.path.append("..")
from myUtils.seg_tools import LABEL2RGB_COLOR_MAP


class ResultSaver:
    def __init__(self, config: Config, dataset: MyDataset):
        self.config = config
        self.dataset = dataset
        self.dataset_config = dataset.dataset_config
        self.class_indicator = self.dataset_config.class_indicator

        self.video_names_list = dataset.video_names_list

        self.output_feature_maps_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "feature_maps")
        self.output_multi_feature_maps_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "multi_feature_maps")
        self.output_probability_prediction_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "probability")
        self.output_affinity_matrix_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name,"affinity_matrix")
        self.output_affinity_attn_matrix_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name,"affinity_matrix_attn")

        self.output_pre_seg_raw_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name,"pre_seg_raw")
        self.output_pre_seg_color_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "pre_seg_color")
        self.output_pre_seg_overlay_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "pre_seg_overlay")
        self.output_gt_seg_color_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "gt_seg_color")
        self.output_gt_seg_overlay_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "gt_seg_overlay")

        self.origin_size = None
        self.origin_size_output = None

    def save(self, return_dict: dict, all_vis=False, te_vis=True):

        b = len(return_dict["x"])
        vis_mark_list = return_dict["vis_mark"]
        for idx in range(b):
            if (vis_mark_list[idx] and te_vis) or all_vis:
                pass
            else:
                continue

            self.origin_size = return_dict["origin_shape"][idx].cpu().numpy().tolist()
            self.origin_size_output = return_dict["origin_shape_output"][idx].cpu().numpy().tolist()

            image_name = return_dict['sample_name'][idx]

            origin_img = return_dict["origin_x"][idx:idx + 1][0]
            origin_img = TF.resize(origin_img, size=self.origin_size, interpolation=TF.InterpolationMode.BICUBIC)
            origin_img = origin_img.permute(1, 2, 0).cpu().numpy()

            gt_seg = return_dict["gt"][idx:idx + 1][0]
            gt_seg = resize_label_map(gt_seg, self.origin_size).unsqueeze(0)

            pre_seg = return_dict["pre_seg"][idx:idx + 1].permute(0, 3, 1, 2)[0].long()
            pre_seg = resize_label_map(pre_seg, self.origin_size).unsqueeze(0)

            if return_dict.get('feats') is not None:
                img = return_dict['feats'][idx].cpu().numpy()

                self.save_intermedia_feature_maps(image_name, img)

            if return_dict.get('feats_multi_block') is not None:
                img = return_dict['feats_multi_block'][idx].cpu().numpy()

                self.save_intermedia_multi_feature_maps(image_name, img)

            if return_dict.get('affinity_matrix') is not None:
                img = return_dict['affinity_matrix'][idx].cpu().numpy()

                self.save_intermedia_affinity(image_name, img)

            if return_dict.get('affinity_matrix_attn') is not None:
                img = return_dict['affinity_matrix_attn'][idx].cpu().numpy()

                self.save_intermedia_affinity_attn(image_name, img)

            if return_dict.get('pro_pre') is not None:
                img = return_dict['pro_pre'][idx].cpu().numpy()

                self.save_pro_pre(image_name, img)

            if return_dict.get('pre_seg_raw') is not None:
                img = return_dict['pre_seg_raw'][idx][..., 0]
                # img = resize_label_map(img, self.origin_size).cpu().numpy()
                img = img.cpu().numpy()
                img = cv2.resize(img, dsize=self.origin_size[::-1], interpolation=cv2.INTER_NEAREST_EXACT)

                self.save_raw_seg(image_name, img)

            self.save_gt_overlay(image_name, gt_seg.permute(1, 2, 0).cpu().numpy(), origin_img)
            self.save_pre_overlay(image_name, pre_seg.permute(1, 2, 0).cpu().numpy(), origin_img)
            self.save_gt_seg(image_name,  gt_seg.permute(1, 2, 0).cpu().numpy())
            self.save_color_result_seg(image_name, pre_seg.permute(1, 2, 0).cpu().numpy())

    def __prepro(self, image, add_border=False):
        image = image.squeeze()
        # assert len(image.shape) == 2, "Image Shape should be (H, W)"

        return image

    def add_border(self, image: np.ndarray):
        if self.dataset_config.prepro_crop and image.shape != self.origin_size:
            if len(image.shape) == 2:
                origin_image = np.zeros(self.origin_size_output)
            else:
                origin_image = np.zeros(self.origin_size_output + [image.shape[-1]])

            x1, y1 = self.dataset_config.valid_field_left_top
            h, w = self.dataset_config.valid_field_size
            h = min(image.shape[0], h)
            w = min(image.shape[1], w)
            origin_image[x1: x1 + h, y1: y1 + w] = image
            image = origin_image
        else:
            pass
        return image

    def get_overlay(self, seg: np.ndarray, image: np.ndarray, alpha=0.4):
        seg = self.__prepro(seg, add_border=True)
        image = self.__prepro(image, add_border=True)

        image_seg = self.get_color_seg(seg).astype(np.uint8)[:, :, ::-1]
        image = (image[:, :, ::-1] * 255).astype(np.uint8)
        image_combined = image.copy()
        image_combined[seg != 0, :] = (image_seg[seg != 0, :] * alpha + image[seg != 0, :] * (1 - alpha)).astype(np.uint8)

        image_combined = self.add_border(image_combined)

        return image_combined

    def save_gt_overlay(self, image_name: str, seg: np.ndarray, image: np.ndarray):
        image_seg = self.get_overlay(seg, image)

        target_path = self.output_gt_seg_overlay_dir + "/" + image_name
        os.makedirs(os.path.split(target_path)[0], exist_ok=True)
        cv2.imwrite(target_path, image_seg)

    def save_pre_overlay(self, image_name: str, seg: np.ndarray, image: np.ndarray):
        image_seg = self.get_overlay(seg, image)

        target_path = self.output_pre_seg_overlay_dir + "/" + image_name
        os.makedirs(os.path.split(target_path)[0], exist_ok=True)
        cv2.imwrite(target_path, image_seg)

    def save_intermedia_feature_maps(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_feature_maps_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for i in range(min(array.shape[-1], 100)):
            target_path = target_path_dir + f"/{i:03}.png"

            img = normalize_01_array(array[..., i]) * 255
            img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

            img = cv2.resize(img, dsize=self.origin_size[::-1], interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_path, img)

        # np.save(target_path, array)

    def save_intermedia_multi_feature_maps(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_multi_feature_maps_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for k in range(min(array.shape[-1], 100)):
            target_path_k = target_path_dir + f"/{k}"
            os.makedirs(target_path_k, exist_ok=True)

            array_k = array[..., k]
            for i in range(min(array_k.shape[-1], 100)):
                target_path = target_path_k + f"/{i:03}.png"

                img = normalize_01_array(array_k[..., i]) * 255
                img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

                img = cv2.resize(img, dsize=self.origin_size[::-1], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(target_path, img)

        # np.save(target_path, array)

    def save_intermedia_affinity(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_affinity_matrix_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for i in range(array.shape[-1]):
            target_path = target_path_dir + f"/{i:03}.png"
            img = array[..., i]
            img = normalize_01_array(img) * 255
            img = img.astype(np.uint8)

            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

            cv2.imwrite(target_path, img)

    def save_intermedia_affinity_attn(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_affinity_attn_matrix_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for i in range(array.shape[-1]):
            target_path = target_path_dir + f"/{i:03}.png"
            img = array[..., i]
            img = normalize_01_array(img) * 255
            img = img.astype(np.uint8)

            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

            cv2.imwrite(target_path, img)

    def save_pro_pre(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_probability_prediction_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for i in range(array.shape[-1]):
            target_path = target_path_dir + f"/{i:03}.png"
            img = array[..., i]
            img = normalize_01_array(img) * 255
            img = img.astype(np.uint8)

            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

            img = cv2.resize(img, dsize=self.origin_size[::-1], interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_path, img)

        # np.save(target_path, array)

    def save_raw_seg(self, image_name: str, image: np.ndarray):

        target_path = self.output_pre_seg_raw_dir + "/" + image_name

        os.makedirs(os.path.split(target_path)[0], exist_ok=True)

        image = self.__prepro(image)

        # for _ in range(image.size // 500000):
        #     image = erosion(image)
        # for _ in range(image.size // 500000):
        #     image = dilation(image)
        #
        # image = remove_small_objects(image, min_size=image.size // 1000)
        # image = remove_small_holes(image, area_threshold=image.size // 100)

        # image_color = (label2rgb(image.astype(np.uint64) + 1, colors=get_color_array())*255).astype(np.uint8)[:,:, ::1]
        image_color = (label2rgb(image.astype(np.uint64) + 1, colors=LABEL2RGB_COLOR_MAP[1:])*255)[:, :, ::-1]

        # scale = image_color.max(axis=-1, keepdims=True)
        # scale[scale == 0] = 255
        # scale = 255.0 / scale

        scale = image_color.max()
        scale = 255.0 / scale if scale != 0 else 1

        image_color *= scale

        image_color = image_color.astype(np.uint8)

        image_color = self.add_border(image_color)

        cv2.imwrite(target_path, image_color)

    def save_gt_seg(self, image_name: str, image: np.ndarray):

        target_path = self.output_gt_seg_color_dir + "/" + image_name

        os.makedirs(os.path.split(target_path)[0], exist_ok=True)

        image = self.__prepro(image)

        image_color = self.get_color_seg(image)

        image_color = self.add_border(image_color)

        cv2.imwrite(target_path, image_color[:, :, ::-1])

    def get_color_seg(self, image):
        image_color = np.zeros(shape=list(image.shape) + [3])

        for class_indicator_name in self.class_indicator:
            class_indicator_i = self.class_indicator[class_indicator_name]
            image_color[image == class_indicator_i[0]] = class_indicator_i[2]

        return image_color

    def save_color_result_seg(self, image_name: str, image: np.ndarray):

        target_path = self.output_pre_seg_color_dir + "/" + image_name

        os.makedirs(os.path.split(target_path)[0], exist_ok=True)

        image = self.__prepro(image)

        # image = resize_label_map(image, self.origin_size)

        image_color = self.get_color_seg(image)

        image_color = self.add_border(image_color)

        cv2.imwrite(target_path, image_color[:, :, ::-1])





