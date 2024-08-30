import copy

import torch.nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from myUtils.config import *
from myUtils.others import *


class MyDataset(Dataset):
    def __init__(self, config: Config, dataset_config: DatasetConfig, data_split_file_path: str, data_aug,
                 train_frame_mode, vis_sample_file_path=None, only_sample=False, train=True):
        super().__init__()

        self.config = config
        self.train = train

        self.data_split_file_path = data_split_file_path
        if self.config.vis:
            self.vis_sample_file_path = vis_sample_file_path
        else:
            self.vis_sample_file_path = None

        self.samples_file_name = self.data_split_file_path.split("/")[-1].split(".")[0]

        self.dataset_config = dataset_config
        self.dataset_name = dataset_config.name + "_" + self.samples_file_name
        self.origin_dataset_name = dataset_config.name

        self.prepro_crop_left_top = dataset_config.valid_field_left_top
        self.prepro_crop_size = dataset_config.valid_field_size
        self.prepro_crop = dataset_config.prepro_crop

        self.frame_left_right = None
        self.test_frame_mode = train_frame_mode

        self.image_dataset = None

        self.data_aug_transforms_x = T.Compose([
        ])
        self.data_aug_transforms_x_gt = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=self.config.model_input_size, scale=(0.75, 1.0)),
        ])

        self.device = self.config.device

        if not only_sample:
            with open(self.data_split_file_path, "r") as f:
                self.samples_names_list = f.read().split("\n")
        else:
            with open(self.vis_sample_file_path, "r") as f:
                self.samples_names_list = f.read().split("\n")

        if self.vis_sample_file_path is not None:
            with open(self.vis_sample_file_path, "r") as f:
                self.vis_samples_names_list = f.read().split("\n")
        else:
            self.vis_samples_names_list = []

        self.vis_samples_names_list = [True if sn in self.vis_samples_names_list else False for sn in
                                       self.samples_names_list]

        self.sample_video_image_dict = {}
        self.image_names_list = [i.split("/")[1] for i in self.samples_names_list]
        self.video_names_list = []
        for sample_name in self.samples_names_list:
            video_name, image_name = sample_name.split("/")
            if not self.sample_video_image_dict.get(video_name):
                if not self.image_dataset:
                    self.sample_video_image_dict[video_name] = {"len": 1, "image_name_list": [image_name]}
                self.video_names_list.append(video_name)
            else:
                if not self.image_dataset:
                    self.sample_video_image_dict[video_name]["len"] += 1
                    self.sample_video_image_dict[video_name]["image_name_list"].append(image_name)

        self.set_extra_frames_train_mode(train_frame_mode)

        self.source_image_path_dict = {
            'x': [f"{self.dataset_config.originImage_dir}/{sample_name_i}" for
                  sample_name_i in self.samples_names_list],
            'gt': [f"{self.dataset_config.groundTruth_dir}/{sample_name_i}" for
                   sample_name_i in self.samples_names_list],
            'video_frame': [sample_name_i.split("/") for
                            sample_name_i in self.samples_names_list]
        }

        video_index_dict = {v: [] for v in self.video_names_list}
        for i in range(len(self.source_image_path_dict["video_frame"])):
            video_index_dict[self.source_image_path_dict["video_frame"][i][0]].append(i)
        self.video_begin_end_dict = {v: [min(video_index_dict[v]), max(video_index_dict[v])] for v in
                                     self.video_names_list}

        self.method_extra_input_process = lambda x: x
        self.extra_input_aug = []

        assert len(
            np.unique([len(i) for i in
                       self.source_image_path_dict.values()])) == 1, "The number of source images are different"

    def __len__(self):
        # return len(self.samples_names_list) - (self.frame_left_right[1] - self.frame_left_right[0])
        if self.image_dataset:
            return len(self.samples_names_list)
        else:
            return sum([d["len"] - (self.frame_left_right[1] - self.frame_left_right[0]) for d in
                        self.sample_video_image_dict.values()])

    def set_extra_frames_train_mode(self, train: bool):
        self.frame_left_right = self.config.train_frame_left_right if train else self.config.test_frame_left_right
        for video_name in self.sample_video_image_dict:
            assert self.frame_left_right[1] - self.frame_left_right[0] + 1 <= self.sample_video_image_dict[video_name][
                "len"]

    def __getitem_image__(self, idx):
        # image_path_list = [i[idx] for i in self.source_image_path_dict]
        return_dict = {
            "dataset_name": self.dataset_name,
            "sample_name": self.samples_names_list[idx],
            "image_idx": idx
        }

        path_x = self.source_image_path_dict["x"][idx]
        path_gt = self.source_image_path_dict["gt"][idx]

        # x input
        x_img = cv2.imread(path_x)[:, :, ::-1]
        x_img = TF.to_tensor(x_img.copy())
        return_dict[f"x_path"] = path_x
        return_dict["origin_shape_output"] = torch.from_numpy(np.array(x_img.shape[1:3]))
        if self.prepro_crop:
            x_img = TF.crop(x_img, *self.prepro_crop_left_top, *self.prepro_crop_size)
        return_dict["origin_shape"] = torch.from_numpy(np.array(x_img.shape[1:3]))
        if len(self.config.model_input_size):
            x_img = TF.resize(x_img, self.config.model_input_size, interpolation=TF.InterpolationMode.BILINEAR)
        return_dict["origin_x"] = torch.clone(x_img.detach())
        x_img = TF.normalize(x_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return_dict["x"] = x_img

        # gt
        if self.dataset_config.available:
            gt_img_ = cv2.imread(path_gt, 0)
            gt_img = np.zeros_like(gt_img_)
            for class_key in self.dataset_config.class_indicator:
                gt_img[gt_img_ == self.dataset_config.class_indicator[class_key][1]] = (
                    self.dataset_config.class_indicator)[class_key][0]
            return_dict[f"gt_path"] = path_gt
            gt_img = torch.from_numpy(gt_img.copy())
            gt_img = gt_img.unsqueeze(-1)
            gt_img = gt_img.permute(2, 0, 1)
            if self.prepro_crop:
                gt_img = TF.crop(gt_img, *self.prepro_crop_left_top, *self.prepro_crop_size)
            if len(self.config.model_input_size):
                gt_img = TF.resize(gt_img, self.config.model_input_size,
                                   interpolation=TF.InterpolationMode.NEAREST_EXACT)
        else:
            gt_img = return_dict["origin_x"].detach().clone().mean(0, keepdim=True).byte()
            return_dict[f"gt_path"] = copy.copy(return_dict["x_path"])
        return_dict["gt"] = gt_img.long()

        return_dict["vis_mark"] = self.vis_samples_names_list[idx]

        return return_dict

    def __getitem_video__(self, origin_idx):
        idx = origin_idx

        video_name, frame_name = self.source_image_path_dict['video_frame'][origin_idx]
        video_begin_idx, video_end_idx = self.video_begin_end_dict[video_name]

        if idx + self.frame_left_right[0] < video_begin_idx:
            idx += video_begin_idx - (idx + self.frame_left_right[0])
        if idx + self.frame_left_right[1] > video_end_idx:
            idx -= idx + self.frame_left_right[1] - video_end_idx

        idx_list = np.arange(self.frame_left_right[0], self.frame_left_right[1] + 1)
        if self.frame_left_right[1] - self.frame_left_right[0] > video_end_idx - video_begin_idx:
            idx = (video_begin_idx + video_end_idx) // 2
            idx_list = idx + idx_list
            idx_list[idx_list < video_begin_idx] = video_begin_idx
            idx_list[idx_list > video_end_idx] = video_end_idx
        else:
            idx_list = idx + idx_list

        idx_list = [(idx_i, idx_i - idx) for idx_i in idx_list]

        return_dict_all_frames = {}

        for idx_i, dict_i in idx_list:
            return_dict = self.__getitem_image__(idx_i)
            return_dict_all_frames[dict_i] = return_dict

        return return_dict_all_frames

    def __getitem__(self, idx):
        return_dict = self.__getitem_video__(idx)
        return return_dict


class DatasetLoader:
    def __init__(self, config: Config, one_sample_train=True, only_sample=False):
        self.config = config

        self.tr_dataset_list = []
        self.tr_no_aug_dataset_list = []
        self.va_dataset_list = []
        self.te_dataset_list = []

        for dataset_config in self.config.dataset_list:
            if self.config.mode == "test" and not dataset_config.available:
                continue
            if not only_sample:
                for txt_file_path in dataset_config.use_for_train:
                    self.tr_dataset_list.append(
                        MyDataset(self.config, dataset_config, txt_file_path, data_aug=True, train_frame_mode=True)
                    )
                for txt_file_path in dataset_config.use_for_val:
                    self.va_dataset_list.append(
                        MyDataset(self.config, dataset_config, txt_file_path, data_aug=False, train_frame_mode=True)
                    )
                for txt_file_path in dataset_config.use_for_test:
                    self.te_dataset_list.append(
                        MyDataset(self.config, dataset_config, txt_file_path, data_aug=False, train_frame_mode=False,
                                  vis_sample_file_path=dataset_config.use_for_vis, train=False)
                    )
            else:
                for txt_file_path in dataset_config.use_for_test:
                    self.te_dataset_list.append(
                        MyDataset(self.config, dataset_config, txt_file_path, data_aug=False, train_frame_mode=False,
                                  vis_sample_file_path=dataset_config.use_for_vis, only_sample=only_sample, train=False)
                    )

        if len(self.tr_dataset_list):
            self.tr_dataset_aug = ConcatDataset(self.tr_dataset_list)
        else:
            self.tr_dataset_aug = self.tr_dataset_list

        if one_sample_train:
            self.va_dataset_list = self.tr_dataset_list
        else:
            pass

        if len(self.va_dataset_list):
            self.va_dataset = ConcatDataset(self.va_dataset_list)
        else:
            self.va_dataset = self.va_dataset_list

        if one_sample_train:
            self.te_dataset = self.tr_dataset_list
        else:
            self.te_dataset = self.te_dataset_list

        self.batch_size_tr = self.config.batch_size if not one_sample_train else 1
        self.batch_size_va = self.config.batch_size if not one_sample_train else 1
        self.batch_size_te = self.config.batch_size_te if not one_sample_train else 1

        if len(self.tr_dataset_aug):
            self.tr_loader = DataLoader(
                self.tr_dataset_aug,
                batch_size=self.batch_size_tr,
                shuffle=not one_sample_train,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
        else:
            self.tr_loader = self.tr_dataset_aug

        if len(self.va_dataset):
            self.va_loader = DataLoader(
                self.va_dataset,
                batch_size=self.batch_size_va,
                shuffle=False,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
        else:
            self.va_loader = self.va_dataset

        self.te_loaders = [
            [DataLoader(
                te_dataset,
                batch_size=self.batch_size_te,
                shuffle=False,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            ), te_dataset] for te_dataset in self.te_dataset
        ]

    def get_datasets_dataloaders(self):
        return self.tr_loader, self.va_loader, self.te_loaders
