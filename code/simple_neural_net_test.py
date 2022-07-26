import torch.nn as nn
import torch.optim as optim
from cifar10_model import CIFAR10TrainingModel
from torchvision import transforms
from models import SNN


if __name__ == "__main__":
    model = CIFAR10TrainingModel(model=SNN(),
                                 learning_rate=1e-3,
                                 optimizer=optim.Adam,
                                 loss_fn=nn.CrossEntropyLoss())
    model.load_and_transform_data(transform_list=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
    model.train(number_of_epochs=2)
    model.plot_acc('Test Plot Accuracy')
    model.plot_loss('Test Plot Loss')
