import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
class TensorboardLogger(object):
    def __init__(self, dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir=dir)
        self.time_s = 0

    def scalar_summary(self, tag, value, t=-1):
        if t == -1:
            self.writer.add_scalar(tag, value, global_step=self.time_s)
        else :
            self.writer.add_scalar(tag, value, global_step=t)

    def histogram_summary(self, tag, tensor):
        self.writer.add_histogram(tag, tensor, global_step=self.time_s)

    def logger_close(self):
        self.writer.close()

    def update(self):
        self.time_s += 1