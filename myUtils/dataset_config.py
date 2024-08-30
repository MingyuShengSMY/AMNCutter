from myUtils.Dict2Class import Dict2Class
import os
from myUtils.others import load_config_file


class DatasetConfig:
    def __init__(self, dataset: dict, dataset_dir: dict, config: Dict2Class, output_root_dir):
        self.output_root_dir = output_root_dir
        self.name = dataset["name"]

        if self.name == "EndoVis2017":
            self.valid_field_left_top = [28, 320]
            self.valid_field_size = [1024, 1280]
        else:
            self.valid_field_left_top = []
            self.valid_field_size = []

        self.prepro_crop = len(self.valid_field_left_top) and len(self.valid_field_size)

        self.input_root_dir = os.path.join(dataset_dir.root, dataset["name"])
        self.inputX_dir = os.path.join(self.input_root_dir, dataset_dir.inputX)
        self.originImage_dir = os.path.join(self.input_root_dir, dataset_dir.originalImage)
        self.groundTruth_dir = os.path.join(self.input_root_dir, dataset_dir.groundTruth)

        self.output_dir = os.path.join(self.output_root_dir, dataset["name"])

        class_indicator_file_path = os.path.join(self.groundTruth_dir, f"{config.task}ClassIndicator.json")
        if os.path.exists(class_indicator_file_path):
            self.class_indicator = Dict2Class(load_config_file(class_indicator_file_path, save_path_name=False)).__dict__
            self.class_n = len(self.class_indicator)
            self.available = True

            os.makedirs(self.input_root_dir, exist_ok=True)
            os.makedirs(self.inputX_dir, exist_ok=True)
            os.makedirs(self.originImage_dir, exist_ok=True)
            os.makedirs(self.groundTruth_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)

        else:
            self.available = False
        self.groundTruth_dir = os.path.join(self.groundTruth_dir, config.task)
        os.makedirs(self.groundTruth_dir, exist_ok=True)
        self.use_for_train = [self.input_root_dir + "/data_split/" + i for i in dataset["use_for_train"]]
        self.use_for_val = [self.input_root_dir + "/data_split/" + i for i in dataset["use_for_val"]]
        self.use_for_test = [self.input_root_dir + "/data_split/" + i for i in dataset["use_for_test"]]
        self.use_for_vis = self.input_root_dir + "/data_split/sample.txt"



