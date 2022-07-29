import torch
import torch.nn as nn
import torch.optim as optim
from cifar10_model import CIFAR10TrainingModel
from torchvision import transforms
from torchvision import models
from torchvision.models import AlexNet_Weights

torch.manual_seed(420)

if __name__ == "__main__":
    alex_net = models.alexnet(AlexNet_Weights)
    # Here we fix the features that will not be trained
    for param in alex_net.features.parameters():
        param.requires_grad = False

    # We also need to modify the final layer of the model. To output a 10-d vector instead of a 1000-d one.
    n_inputs = alex_net.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, 10)
    alex_net.classifier[6] = last_layer
    alex_net_model = CIFAR10TrainingModel(model=alex_net,
                                          learning_rate=1e-4,
                                          optimizer=optim.Adam,
                                          loss_fn=nn.CrossEntropyLoss())

    alex_net_model.load_and_transform_data(transform_list=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

    n_epochs = 5
    alex_net_model.overfit_single_batch(n_epochs)

