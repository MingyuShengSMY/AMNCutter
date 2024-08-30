import sys
import time

from myUtils.config import Config
from myUtils.Log import CsvLog

from myUtils.others import *
from sklearn.metrics import confusion_matrix as get_confusion_matrix

sys.path.append("..")
from dataloader import MyDataset


class Metrics:
    def __init__(self, model, config: Config, dataset_list: list[MyDataset], save=True):
        self.model = model
        self.config = config

        self.dataset_list = dataset_list
        self.dataset_name_dict = {dataset.dataset_name: dataset for dataset in dataset_list}

        self.intersection_sum_class_dict = {}
        self.union_sum_class_dict = {}
        self.class_pixel_num_dict = {}
        self.iou_class_dict = {}
        self.dice_class_dict = {}
        self.total_pixel_num_no_background = {}
        self.total_pixel_num = {}

        self.image_num = {}
        self.time_cost = {}

        self.img_miou = {}
        self.sample_name = {}
        self.cluster_k = {}
        self.eigen_tree = {}

        self.class_count = {}

        self.header = [
            "DatasetName",
            "mIoU",
            "Acc",
            "mDice",
            "SPF",
            "FPS",
            "Params (M)"
        ]

        self.all_metrics_dict = {}
        self.all_result_record_dict = {}

        self.dataset_name_list = [dataset.dataset_name for dataset in dataset_list]
        for dataset_name in self.dataset_name_list:
            self.all_metrics_dict[dataset_name] = {k: 0.0 for k in self.header}
            self.all_metrics_dict[dataset_name]["DatasetName"] = dataset_name
            self.all_metrics_dict[dataset_name]["Params (M)"] = count_parameters(self.model) * 1e-6

            self.all_result_record_dict[dataset_name] = {}

            class_indicator = self.dataset_name_dict[dataset_name].dataset_config.class_indicator

            self.all_result_record_dict[dataset_name]["class_indicator"] = class_indicator

            self.all_result_record_dict[dataset_name]["labels_list"] = [c[0] for c in class_indicator.values()]

            self.all_result_record_dict[dataset_name]["class_idx"] = {c: class_indicator[c][0] for c in class_indicator}

            class_num = len(class_indicator)

            # self.all_result_record_dict[dataset_name]["class_count"] = {c: 0 for c in class_indicator}

            self.all_result_record_dict[dataset_name]["confusion_matrix"] = np.zeros(shape=(class_num, class_num))

            self.all_result_record_dict[dataset_name]["time_cost"] = 0

            self.all_result_record_dict[dataset_name]["image_num"] = 0

            self.all_result_record_dict[dataset_name]["img_miou"] = []

            self.all_result_record_dict[dataset_name]["sample_name"] = []

        self.save = save
        if self.save:
            self.csv_logger = CsvLog(self.config.log_dir, f"metrics", header=self.header)

        self.start_time = 0
        self.end_time = 0

        self.dataset_name_list = list(self.dataset_name_dict.keys())
        longest_name_length = max([len(i) for i in self.dataset_name_list]) + 1

        self.print_format_count = [max(len(k) + 1, 12) for k in self.header]
        self.print_format_count[0] = max(longest_name_length, self.print_format_count[0])
        self.print_format_count[0] = max(self.print_format_count[0], len(max(self.dataset_name_list)) + 1)

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def metrics_update(self, return_dict):
        time_cost = return_dict["time_cost"]
        dataset_name = return_dict["dataset_name"][0]
        sample_names = return_dict["sample_name"]

        pre_seg = return_dict['pre_seg']
        gt_seg = return_dict['gt']

        b = len(gt_seg)

        self.all_result_record_dict[dataset_name]["time_cost"] += time_cost
        self.all_result_record_dict[dataset_name]["image_num"] += b
        self.all_result_record_dict[dataset_name]["sample_name"] += sample_names

        labels_list = self.all_result_record_dict[dataset_name]["labels_list"]

        for i in range(b):
            pre_seg_i = pre_seg[i].flatten().cpu().numpy()
            gt_seg_i = gt_seg[i].flatten().cpu().numpy()

            confusion_matrix: np.ndarray = get_confusion_matrix(y_true=gt_seg_i, y_pred=pre_seg_i, labels=labels_list)

            pre_cm_sum = confusion_matrix.sum(axis=0)
            gt_cm_sum = confusion_matrix.sum(axis=1)
            inter_cm_sum = confusion_matrix.diagonal()

            union_cm_sum = gt_cm_sum + pre_cm_sum - inter_cm_sum
            union_cm_sum[union_cm_sum == 0] = 1

            img_iou_array = inter_cm_sum / union_cm_sum

            img_miou = img_iou_array.mean(where=gt_cm_sum != 0)

            self.all_result_record_dict[dataset_name]["confusion_matrix"] += confusion_matrix
            self.all_result_record_dict[dataset_name]["img_miou"].append(img_miou)

    def get_metrics(self, dataset_name):
        self.all_metrics_dict[dataset_name]['SPF'] = (self.all_result_record_dict[dataset_name]["time_cost"] /
                                                      self.all_result_record_dict[dataset_name]["image_num"])
        self.all_metrics_dict[dataset_name]['FPS'] = self.all_result_record_dict[dataset_name]["image_num"] / self.all_result_record_dict[dataset_name]["time_cost"]

        confusion_matrix: np.ndarray = self.all_result_record_dict[dataset_name]['confusion_matrix']

        pre_cm_sum = confusion_matrix.sum(axis=0)
        gt_cm_sum = confusion_matrix.sum(axis=1)
        inter_cm_sum = confusion_matrix.diagonal()

        union_cm_sum = gt_cm_sum + pre_cm_sum - inter_cm_sum
        union_cm_sum[union_cm_sum == 0] = 1

        union_dice_cm_sum = gt_cm_sum + pre_cm_sum
        union_dice_cm_sum[union_dice_cm_sum == 0] = 1

        iou_array = inter_cm_sum / union_cm_sum
        dice_array = inter_cm_sum * 2 / union_dice_cm_sum

        miou = iou_array.mean(where=gt_cm_sum != 0)
        mdice = dice_array.mean(where=gt_cm_sum != 0)
        acc = inter_cm_sum.sum() / gt_cm_sum.sum()

        self.all_metrics_dict[dataset_name]['mIoU'] = miou
        self.all_metrics_dict[dataset_name]['mDice'] = mdice
        self.all_metrics_dict[dataset_name]['Acc'] = acc

        if self.save:
            self.csv_logger.log_a(self.all_metrics_dict[dataset_name])

        return self.all_metrics_dict[dataset_name]

    def get_all_metrics(self):

        total_sample_count = sum(self.all_result_record_dict[dn]["image_num"] for dn in self.dataset_name_list)

        ratio_array = []
        spf_array = []
        fps_array = []
        miou_array = []
        mdice_array = []
        acc_array = []
        sample_list = []
        miou_list = []
        confusion_matrix_dict = {}
        for dataset_name in self.dataset_name_dict:
            dataset_img_num = self.all_result_record_dict[dataset_name]["image_num"]
            ratio = dataset_img_num / total_sample_count

            ratio_array.append(ratio)
            spf_array.append(self.all_metrics_dict[dataset_name]['SPF'])
            fps_array.append(self.all_metrics_dict[dataset_name]['FPS'])
            miou_array.append(self.all_metrics_dict[dataset_name]['mIoU'])
            mdice_array.append(self.all_metrics_dict[dataset_name]['mDice'])
            acc_array.append(self.all_metrics_dict[dataset_name]['Acc'])

            sample_list += self.all_result_record_dict[dataset_name]['sample_name']
            miou_list += self.all_result_record_dict[dataset_name]['img_miou']

            confusion_matrix_dict[dataset_name] = {
                "confusion_matrix": self.all_result_record_dict[dataset_name]["confusion_matrix"],
                "class_idx": self.all_result_record_dict[dataset_name]["class_idx"],
                }

        ratio_array = np.array(ratio_array)
        spf_array = np.array(spf_array)
        fps_array = np.array(fps_array)
        miou_array = np.array(miou_array)
        mdice_array = np.array(mdice_array)
        acc_array = np.array(acc_array)

        self.all_metrics_dict["Avg"] = {k: 0.0 for k in self.header}
        self.all_metrics_dict["Avg"].update({
            "DatasetName": "Avg",
            "Params (M)": count_parameters(self.model) * 1e-6
        })

        self.all_metrics_dict["Avg"]["SPF"] = (spf_array * ratio_array).sum()
        self.all_metrics_dict["Avg"]["mIoU"] = (miou_array * ratio_array).sum()
        self.all_metrics_dict["Avg"]["mDice"] = (mdice_array * ratio_array).sum()
        self.all_metrics_dict["Avg"]["Acc"] = (acc_array * ratio_array).sum()

        self.all_metrics_dict["Avg"]['FPS'] = (fps_array * ratio_array).sum()

        self.all_metrics_dict["Std"] = {k: 0.0 for k in self.header}
        self.all_metrics_dict["Std"].update({
            "DatasetName": "Std",
            "Params (M)": 0
        })

        self.all_metrics_dict["Std"]["SPF"] = np.sqrt((np.power((spf_array - self.all_metrics_dict["Avg"]["SPF"]), 2) * ratio_array).sum())
        self.all_metrics_dict["Std"]["FPS"] = np.sqrt((np.power((fps_array - self.all_metrics_dict["Avg"]["FPS"]), 2) * ratio_array).sum())
        self.all_metrics_dict["Std"]["mIoU"] = np.sqrt((np.power((miou_array - self.all_metrics_dict["Avg"]["mIoU"]), 2)* ratio_array).sum())
        self.all_metrics_dict["Std"]["mDice"] = np.sqrt((np.power((mdice_array - self.all_metrics_dict["Avg"]["mDice"]), 2) * ratio_array).sum())
        self.all_metrics_dict["Std"]["Acc"] = np.sqrt((np.power((acc_array - self.all_metrics_dict["Avg"]["Acc"]), 2) * ratio_array).sum())

        if self.save:
            self.csv_logger.log_a(self.all_metrics_dict["Avg"])
            self.csv_logger.log_a(self.all_metrics_dict["Std"])

            img_name_miou = {img_name: img_miou for img_name, img_miou in zip(sample_list, miou_list)}

            torch.save(img_name_miou, self.config.log_dir + "/" + "single_img_miou.bin")

            torch.save(confusion_matrix_dict, self.config.log_dir + "/" + "confusion_matrix.bin")

        return self.all_metrics_dict["Avg"], self.all_metrics_dict["Std"]

    def print_header(self):
        print_string = [f"{s:>{self.print_format_count[i]}}" for i, s in enumerate(self.header)]
        print_string = "|".join(print_string)
        print(print_string)

    def print_dataset_metric(self, dataset: MyDataset):
        metrics_dict = self.get_metrics(dataset.dataset_name)
        metric_values_array = np.array(list(metrics_dict.values())[1:])
        metric_values_array = metric_values_array.round(4)
        metrics_values = [metrics_dict["DatasetName"]] + metric_values_array.astype(str).tolist()

        print_string = [f"{s:>{self.print_format_count[i]}}" for i, s in enumerate(metrics_values)]
        print_string = " ".join(print_string)
        print(print_string)

    def print_all_metric(self):
        metrics_dict_avg, metrics_dict_std = self.get_all_metrics()

        metrics_dict = metrics_dict_avg
        metric_values_array = np.array(list(metrics_dict.values())[1:])
        metric_values_array = metric_values_array.round(4)
        metrics_values = [metrics_dict["DatasetName"]] + metric_values_array.astype(str).tolist()

        print_string = [f"{s:>{self.print_format_count[i]}}" for i, s in enumerate(metrics_values)]
        print_string = " ".join(print_string)
        print(print_string)

        metrics_dict = metrics_dict_std
        metric_values_array = np.array(list(metrics_dict.values())[1:])
        metric_values_array = metric_values_array.round(4)
        metrics_values = [metrics_dict["DatasetName"]] + metric_values_array.astype(str).tolist()

        print_string = [f"{s:>{self.print_format_count[i]}}" for i, s in enumerate(metrics_values)]
        print_string = " ".join(print_string)
        print(print_string)





