from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # for video dataset, the number of extra frames, [-1, 1] means 3 frames used including the mid one
        self.train_frame_left_right = [0, 0]
        self.test_frame_left_right = [0, 0]
        self.loss_detail_key_list = ["Loss_Sum"]

    def get_loss(self, return_dict):
        pass

    def get_output(self, return_dict):
        pass

    def get_seg(self, return_dict, vis=False):
        pass

    def iter_before_operate(self, epoch, ite, max_ite):
        pass

    def iter_after_operate(self, epoch, ite, max_ite):
        pass

    def epoch_before_operate(self, epoch, data_loader):
        pass

    def epoch_after_operate(self, epoch):
        pass

    def training_vis(self, return_dict, img_idx):
        pass

    def test_vis_save(self):
        pass