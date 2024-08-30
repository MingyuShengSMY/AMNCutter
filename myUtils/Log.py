import os

import numpy as np
from tqdm import tqdm

from myUtils.config import Config


class LossLogger:
    def __init__(self, config: Config):
        self.config = config

        self.loss_detail_key_list = self.config.loss_detail_key_list

        self.log_header = ["Epoch", "Min_Va_Loss"]

        for loss_verbose_i in self.loss_detail_key_list:
            self.log_header.append(f"Tr_{loss_verbose_i}")
            self.log_header.append(f"Va_{loss_verbose_i}")

        self.print_format_num = [len(self.log_header[0])] + [max(len(i) + 1, 12) for i in self.log_header[1:]]

        self.csv_logger_loss = None
        self.npy_logger_loss = None

    def log_file_init(self, file_name="loss"):
        self.csv_logger_loss = CsvLog(self.config.log_dir, "loss", header=self.log_header)
        self.npy_logger_loss = NpyLog(self.config.log_dir, "loss", header=self.log_header)

    def log_print_head(self):
        print_string = [f"{str(self.log_header[i]):>{self.print_format_num[i]}}"
                        for i in range(len(self.log_header))]
        print_string = "|".join(print_string)
        print(print_string)

    def log_print_loss(self, epoch, log_losses_list):
        print_string = [epoch] + np.array(log_losses_list[1:]).round(6).tolist()
        print_string = [f"{str(print_string[i]):>{self.print_format_num[i]}}"
                        for i in range(len(self.log_header))]
        print_string = " ".join(print_string)
        tqdm.write(print_string)

    def get_log_loss_dict(self, epoch, min_va_loss, tr_loss_detail, va_loss_detail):
        log_losses_dict = {k: 0 for k in self.log_header}
        log_losses_dict["Epoch"] = epoch
        log_losses_dict["Min_Va_Loss"] = min_va_loss
        log_losses_list = [epoch, min_va_loss]
        if len(self.loss_detail_key_list) > 1:
            for loss_key_i in self.loss_detail_key_list:
                log_losses_list.append(tr_loss_detail[loss_key_i])
                log_losses_list.append(va_loss_detail[loss_key_i])
                log_losses_dict[f"Tr_{loss_key_i}"] = tr_loss_detail[loss_key_i]
                log_losses_dict[f"Va_{loss_key_i}"] = va_loss_detail[loss_key_i]

        return log_losses_dict, log_losses_list

    def log_in_file(self, log_losses_dict, log_losses_list):
        self.csv_logger_loss.log_a(log_losses_dict)
        self.npy_logger_loss.log_a(log_losses_list)


class TextLog:
    def __init__(self, log_dir, log_file_name):
        self.log_path = os.path.join(log_dir, log_file_name + ".txt")
        with open(self.log_path, "w") as f:
            pass

    def log_a(self, string):
        with open(self.log_path, "a") as f:
            f.write(string)

    def log_w(self, string):
        with open(self.log_path, "w") as f:
            f.write(string)


class CsvLog:
    def __init__(self, log_dir, log_file_name, header: list):

        self.log_path = os.path.join(log_dir, log_file_name + ".csv")

        self.header = header
        self.header = np.array(self.header)
        self.header = self.header.flatten().astype(str)
        self.column_num = len(self.header)

        header = {k: k for k in header}

        self.log_w(header)

    def dict2text(self, array: dict):
        v_list = []
        for k in self.header:
            v = str(array[k])
            v_list.append(v)

        text = ",".join(v_list) + "\n"
        return text

    def log_a(self, array: dict):
        with open(self.log_path, "a") as f:
            f.write(self.dict2text(array))

    def log_w(self, array: dict):
        with open(self.log_path, "w") as f:
            f.write(self.dict2text(array))


class NpyLog:
    def __init__(self, log_dir, log_file_name, header: list):

        self.log_path = os.path.join(log_dir, log_file_name + ".npy")

        self.header = header

        self.column_num = len(self.header)

        self.np_log = []
        self.np_log.append(self.header)

    def log_a(self, array: list):
        self.np_log.append(array)
        self.save()

    def save(self):
        np.save(self.log_path, np.array(self.np_log), allow_pickle=True)

