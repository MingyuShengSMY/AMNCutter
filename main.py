import argparse
import time

import torch.optim
from tqdm import *

from myUtils.EarlyStopping import EarlyStopping
from myUtils.Log import LossLogger
from myUtils.config import *
from myUtils.metrics import Metrics
from myUtils.model_config_list import SUP_METHOD_LIST
from myUtils.others import *
from myUtils.model_selector import *
from dataloader import DatasetLoader, MyDataset
from myUtils.others import load_config_file
from myUtils.result_saver import ResultSaver
from myUtils.seg_tools import SegTool

# RUN_CHECK = True
RUN_CHECK = False

# ONE_SAMPLE_TRAIN = True
ONE_SAMPLE_TRAIN = False

LOG_SAVE = True
# LOG_SAVE = False

# ONLY_SAMPLE_IMG = True
ONLY_SAMPLE_IMG = False

if ONLY_SAMPLE_IMG:
    LOG_SAVE = False

SAVE_OUTPUT_SEG = True
# SAVE_OUTPUT_SEG = False

# TRAINING_VIS_ITER = True
TRAINING_VIS_ITER = False

# TRAINING_VIS_EPOCH = True
TRAINING_VIS_EPOCH = False


class Method:
    def __init__(self, config: Config):
        self.model = None
        self.mode = config.mode
        self.optimizer = None
        self.config = config
        self.pre_load_dict = None

        self.learning_rate = self.config.learning_rate
        self.lr_beta1 = self.config.lr_beta1
        self.lr_beta2 = self.config.lr_beta2

        self.early_stopper = EarlyStopping(self.config.early_stop_patience)

        self.device = self.config.device

        self.model = model_selector(self.config).to(self.device)

        self.config.train_frame_left_right = self.model.train_frame_left_right
        self.config.test_frame_left_right = self.model.test_frame_left_right
        self.config.loss_detail_key_list = self.model.loss_detail_key_list

        self.loss_detail_key_list = self.model.loss_detail_key_list

        print("Parameters: ", count_parameters(self.model) * 1e-6)

        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.trained_module_list.parameters(), lr=self.config.learning_rate, betas=(self.config.lr_beta1, self.config.lr_beta2), weight_decay=self.config.opt_weight_decay)
        elif self.config.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.trained_module_list.parameters(),
                                              lr=self.config.learning_rate,
                                              betas=(self.config.lr_beta1, self.config.lr_beta2),
                                              weight_decay=self.config.opt_weight_decay)
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.trained_module_list.parameters(),
                                              lr=self.config.learning_rate,
                                              momentum=self.config.lr_beta1)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        if self.config.decay == "cos":
            self.lr_decay_mark = True
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.config.epoch_num)
        else:
            self.lr_decay_mark = False
            self.scheduler = None

        os.makedirs(self.config.model_save_dir, exist_ok=True)

        self.load_model_mark = len(self.config.model_load_from) and self.config.load_model_mark

        if self.load_model_mark:
            self.pre_load_dict = torch.load(self.config.model_load_from)
            if self.pre_load_dict.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(self.pre_load_dict.get("optimizer_state_dict"))
            if self.pre_load_dict.get("model_state_dict"):
                self.model.load_state_dict(self.pre_load_dict.get("model_state_dict"))
            if self.pre_load_dict.get("early_stop_patience"):
                self.early_stopper.epoch_patience = self.pre_load_dict.get("early_stop_patience")
            if self.pre_load_dict.get("min_va_loss"):
                self.early_stopper.min_loss = self.pre_load_dict.get("min_va_loss")
            if self.pre_load_dict.get("early_stop_count"):
                self.early_stopper.stop_count = self.pre_load_dict.get("early_stop_count")
        else:
            pass

        if ONE_SAMPLE_TRAIN:
            self.config.epoch_num = 100000000

        self.data_loader = DatasetLoader(self.config, one_sample_train=ONE_SAMPLE_TRAIN, only_sample=ONLY_SAMPLE_IMG)

        self.tr_loader, self.va_loader, self.te_loaders = self.data_loader.get_datasets_dataloaders()

        self.loss_logger = LossLogger(self.config)

        self.global_match = False
        self.unsup_method = self.config.method_name not in SUP_METHOD_LIST

    def __train_or_val(self, epoch, data_loader, train):
        if train:
            self.model.train()
            self.model.requires_grad_(True)
            self.model.frozen_module_list.eval()
            self.model.frozen_module_list.requires_grad_(False)

        else:
            self.model.eval()
            self.model.requires_grad_(False)

        batch_range = tqdm(
            data_loader,
            position=1,
            desc=("Training" if train else "Validating") + f" Epoch-{epoch}",
            leave=False
        )

        iter_max = len(data_loader)

        mean_loss_detail = {k: 0 for k in self.loss_detail_key_list}

        sample_batch = None if ONE_SAMPLE_TRAIN else data_loader.dataset.datasets[0].__getitem__(0)

        n = 0
        iter_i = 0
        max_iter = len(data_loader)
        for batch_dict in batch_range:
            batch_size = len(batch_dict[0]["x"])
            batch_dict["epoch"] = epoch
            batch_dict["iter"] = iter_i
            batch_dict["max_iter"] = iter_max

            if iter_i == 0 and ONE_SAMPLE_TRAIN:
                sample_batch = batch_dict.copy()

            self.model.iter_before_operate(epoch, iter_i, iter_max)

            loss_detail = self.model.get_loss(batch_dict)

            loss = loss_detail.get('Loss_Grad')
            if loss is None:
                loss = loss_detail.get("Loss_Sum")

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.lr_decay_mark:
                    self.scheduler.step(epoch + iter_i / max_iter)

            self.model.iter_after_operate(epoch, iter_i, iter_max)

            iter_i += 1
            n += batch_size
            for k in mean_loss_detail:
                mean_loss_detail[k] += loss_detail[k] * batch_size

            if TRAINING_VIS_ITER and train:
                with torch.no_grad():
                    vis_batch_dict = sample_batch
                    self.model.training_vis(return_dict=vis_batch_dict, img_idx=epoch * iter_max + iter_i)

            if RUN_CHECK or ONE_SAMPLE_TRAIN:
                break

        if TRAINING_VIS_EPOCH and train:
            with torch.no_grad():
                vis_batch_dict = sample_batch
                self.model.training_vis(return_dict=vis_batch_dict, img_idx=epoch * iter_max + iter_i)

        for k in mean_loss_detail:
            mean_loss_detail[k] /= n
            if torch.is_tensor(mean_loss_detail[k]):
                mean_loss_detail[k] = mean_loss_detail[k].item()
        return mean_loss_detail

    def train(self):
        if LOG_SAVE:
            self.loss_logger.log_file_init()

        min_va_loss = np.inf

        start_epoch = 0
        if self.pre_load_dict:
            if self.pre_load_dict.get("epoch"):
                start_epoch = self.pre_load_dict.get("epoch") + 1
            if self.pre_load_dict.get("min_va_loss"):
                min_va_loss = self.pre_load_dict.get("min_va_loss")

        epoch_range = tqdm(
            range(start_epoch, self.config.epoch_num),
            leave=False,
            position=0,
            desc="Training"
        )

        if self.config.verbose:
            self.loss_logger.log_print_head()

        for epoch in epoch_range:

            self.model.epoch_before_operate(epoch, self.tr_loader)

            tr_loss_detail = self.__train_or_val(epoch, self.tr_loader, train=True)
            va_loss_detail = self.__train_or_val(epoch, self.va_loader, train=False)

            self.model.epoch_after_operate(epoch)

            if va_loss_detail['Loss_Sum'] < min_va_loss:
                min_va_loss = va_loss_detail['Loss_Sum']

                torch.save(
                    obj={
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config.config
                    },
                    f=self.config.model_save_dir + "/trained_model"
                )

            log_losses_dict, log_losses_list = self.loss_logger.get_log_loss_dict(epoch, min_va_loss, tr_loss_detail, va_loss_detail)

            if LOG_SAVE:
                self.loss_logger.log_in_file(log_losses_dict, log_losses_list)

            if epoch % self.config.checkpoint_per_epoch_num == 0 and not RUN_CHECK and not ONE_SAMPLE_TRAIN:
                torch.save(
                    obj={
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "min_va_loss": min_va_loss,
                        "epoch": epoch,
                        "early_stop_patience": self.early_stopper.epoch_patience,
                        "early_stop_count": self.early_stopper.stop_count,
                        "config": self.config.config
                    },
                    f=self.config.model_save_dir + "/checkpoint"
                )

            if self.config.verbose:
                self.loss_logger.log_print_loss(epoch, log_losses_list)

            if self.early_stopper.early_stop_check(va_loss_detail['Loss_Sum']) and not ONE_SAMPLE_TRAIN:
                tqdm.write("Early Stopped")
                break

            if RUN_CHECK:
                break

        print("Training Done")

    def __test_dataset(self, data_loader, dataset: MyDataset, measurer: Metrics):
        batch_range_cluster_class_map = tqdm(
            data_loader,
            position=1,
            desc=f"Hungarian Matching on {dataset.dataset_name}",
            leave=False
        )

        batch_range_vis = tqdm(
            data_loader,
            position=1,
            desc=f"Testing on {dataset.dataset_name}",
            leave=False
        )

        result_saver = ResultSaver(self.config, dataset)

        seg_tool = SegTool(self.config, dataset)

        if self.global_match and self.unsup_method:
            seg_tool.hungarian_match_init()
            for batch_dict in batch_range_cluster_class_map:
                batch_dict = self.model.get_seg(batch_dict)
                seg_tool.hungarian_match_record(batch_dict[0])
                if RUN_CHECK or ONE_SAMPLE_TRAIN:
                    break
            seg_tool.hungarian_get_match()

        for batch_dict in batch_range_vis:
            s_timer = time.time()
            batch_dict = self.model.get_seg(batch_dict, vis=SAVE_OUTPUT_SEG and self.config.vis)
            e_timer = time.time()

            batch_dict[0]["time_cost"] = e_timer - s_timer

            i = 0

            seg_tool.seg_postprocessing(return_dict_i=batch_dict[i], match=self.unsup_method, single_match=not self.global_match)

            result_saver.save(batch_dict[i], all_vis=SAVE_OUTPUT_SEG and self.config.vis and RUN_CHECK, te_vis=SAVE_OUTPUT_SEG and self.config.vis)

            if dataset.samples_file_name != "sample":
                measurer.metrics_update(batch_dict[i])

            if RUN_CHECK or ONE_SAMPLE_TRAIN:
                break

    def test(self):
        self.model.eval()
        self.model.requires_grad_(False)

        self.model.test_vis_save()

        dataloader_list = self.te_loaders

        metric_dataset_list = [i[1] for i in dataloader_list if i[1].samples_file_name != "sample" or ONLY_SAMPLE_IMG]

        measurer = Metrics(self.model, self.config, metric_dataset_list, save=LOG_SAVE)

        measurer.print_header()

        for te_loader, te_dataset in dataloader_list:
            self.__test_dataset(te_loader, te_dataset, measurer)

            if te_dataset.samples_file_name != "sample":
                measurer.print_dataset_metric(te_dataset)

        measurer.print_all_metric()


def main():
    args = parser.parse_args()

    config = load_config_file(args.config_file)

    config["mode"] = args.mode
    config["vis"] = args.vis

    config = Dict2Class(config)

    config = Config(config)

    if config.no_available_dataset:
        print("No available dataset")
        return

    print(f"seed: {config.random_seed}")
    seed_everything(config.random_seed)

    if args.mode == 'test':
        config.load_model_mark = True

    method = Method(config)

    if args.mode == 'train':
        method.train()
    else:
        method.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='path/to/target/config_file.json')
    parser.add_argument('--mode', help='train or test')
    parser.add_argument('--vis', default=False, action="store_true", help='visualization')

    main()
