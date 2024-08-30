import numpy as np


class EarlyStopping:
    def __init__(self, epoch_patience, stop_count=0, min_loss=np.inf):
        self.epoch_patience = epoch_patience
        self.stop_count = stop_count
        self.min_loss = min_loss

    def early_stop_check(self, loss):
        if loss > self.min_loss:
            self.stop_count += 1
        else:
            self.min_loss = loss
            self.stop_count = 0

        return self.stop_count >= self.epoch_patience
