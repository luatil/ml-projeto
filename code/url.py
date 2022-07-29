from cifar10_model import CIFAR10TrainingModel
from models import SNN, LeNet
import torch.optim as optim
import torch.nn as nn


model = CIFAR10TrainingModel(model=LeNet(),
                             learning_rate=1e-3,
                             optimizer=optim.Adam,
                             loss_fn=nn.CrossEntropyLoss())

model.load_model('../model/lenet.pth')

deer_url = 'https://c8.alamy.com/comp/DYC06A/hornless-reindeer-at-zoo-DYC06A.jpg'
dog_url = 'https://images.unsplash.com/photo-1587402092301-725e37c70fd8?ixlib=rb-1.2.1&ixid' \
          '=MnwxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGRvZ3N8ZW58MHx8MHx8&w=1000&q=80 '
print(model.prediction_from_url(deer_url))
