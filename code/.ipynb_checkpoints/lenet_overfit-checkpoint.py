import torch
import torch.nn as nn
import torch.optim as optim
from cifar10_model import CIFAR10TrainingModel
from torchvision import transforms
from models import LeNet

torch.manual_seed(420)

if __name__ == "__main__":
    model = CIFAR10TrainingModel(model=LeNet(),
                                 learning_rate=1e-3,
                                 optimizer=optim.Adam,
                                 loss_fn=nn.CrossEntropyLoss())

    model.load_and_transform_data(transform_list=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

    n_epochs = 20
    # print(model.get_number_of_trainable_parameters())
    model.overfit_single_batch(n_epochs)
