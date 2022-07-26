import torch
from models import LeNet

torch.manual_seed(420)

if __name__ == "__main__":
    n_epochs = 10
    lenet_model = LeNet()
