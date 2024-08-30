import os
import torch

from myUtils.Dict2Class import Dict2Class
from myUtils.dataset_config import DatasetConfig
from myUtils.model_config_list import SUP_METHOD_LIST


class Config:
    def __init__(self, config: Dict2Class):
        self.config = config

        try:
            self.mode = self.config.mode
        except AttributeError:
            self.mode = "test"

        self.config_file_path = self.config.config_file_path
        self.config_file_name = self.config.config_file_name

        self.config_file_path_name = "/".join(self.config_file_path.split("/")[1:-1])

        self.output_root_dir = os.path.join("outputs", self.config_file_path_name, self.config_file_name)
        if self.mode == "test":
            os.makedirs(self.output_root_dir, exist_ok=True)

        self.method_name = self.config.method_name
        self.method_config = self.config.method_config_dict.__dict__

        self.task = self.config.task

        self.dataset_dir = self.config.dataset_dir
        self.dataset_list = [DatasetConfig(i, self.dataset_dir, self.config, self.output_root_dir) for i in self.config.dataset_list]
        self.no_available_dataset = False
        if self.method_name in SUP_METHOD_LIST or (not self.method_config.get("cluster_k") or self.method_config.get("cluster_k") < 0):
            class_n_list = [dc.class_n for dc in self.dataset_list if dc.available]
            if len(class_n_list):
                self.method_config["cluster_k"] = max(class_n_list)
            else:
                self.no_available_dataset = True

        self.log_dir = os.path.join(self.output_root_dir, self.config.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.load_model_mark = self.config.load_model_mark

        if len(self.config.model_load_from) == 0:
            load_path = os.path.join(self.config_file_path_name, self.config_file_name, "trained_model")
        else:
            load_path = self.config.model_load_from

        if len(self.config.model_save_dir) == 0:
            save_path = os.path.join(self.config_file_path_name, self.config_file_name)
        else:
            save_path = self.config.model_save_dir

        self.model_load_from = os.path.join("saved_models", load_path)
        self.model_save_dir = os.path.join("saved_models", save_path)

        self.model_input_size = tuple(self.config.model_input_size)

        self.epoch_num = self.config.epoch_num
        self.checkpoint_per_epoch_num = self.config.checkpoint_per_epoch_num
        self.early_stop_patience = self.config.early_stop_patience

        self.batch_size = self.config.batch_size
        try:
            self.batch_size_te = self.config.batch_size_te
        except AttributeError:
            self.batch_size_te = self.batch_size
        self.num_workers = self.config.num_workers

        self.optimizer = self.config.optimizer
        self.learning_rate = self.config.learning_rate
        self.lr_beta1 = self.config.lr_beta1
        self.lr_beta2 = self.config.lr_beta2
        try:
            self.decay = self.config.decay
        except AttributeError:
            self.decay = "none"
        try:
            self.epoch_update = self.config.epoch_update
        except AttributeError:
            self.epoch_update = False

        try:
            self.opt_weight_decay = self.config.weight_decay
        except AttributeError:
            self.opt_weight_decay = 0

        self.early_stop_patience = self.config.early_stop_patience

        self.random_seed = self.config.random_seed

        self.gpu_mark = self.config.gpu_mark
        self.device = torch.device("cuda") if self.gpu_mark and torch.cuda.is_available() else torch.device('cpu')

        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(self.random_seed)

        self.torch_generator_cpu = torch.Generator(device=torch.device('cpu'))
        self.torch_generator_cpu.manual_seed(self.random_seed)

        self.verbose = self.config.verbose

        self.vis = self.config.vis

        self.train_frame_left_right = None
        self.test_frame_left_right = None
        self.loss_detail_key_list = None


