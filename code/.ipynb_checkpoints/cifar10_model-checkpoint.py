import requests
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets
import torch
from torchvision.transforms import transforms


class CIFAR10TrainingModel:
    def __init__(self, model, learning_rate, optimizer, loss_fn):

        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.data_path = '../data/'

        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = loss_fn

        self.data = None
        self.train_set = None
        self.val_set = None
        self.overfit_set = None
        self.val_loader = None
        self.train_loader = None
        self.overfit_loader = None

        self.trained_epochs = 0
        self.validation_loss_hist = []
        self.validation_acc_hist = []
        self.training_loss_hist = []
        self.training_acc_hist = []

    def load_and_transform_data(self, transform_list):
        self.data = datasets.CIFAR10(self.data_path, train=True, download=True,
                                     transform=transform_list)
        self.train_set, self.val_set, self.overfit_set = torch.utils.data.random_split(self.data,
                                                                                       [40000, 10000 - 100, 100])
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=64,
                                                      shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64,
                                                        shuffle=True)
        self.overfit_loader = torch.utils.data.DataLoader(self.overfit_set, batch_size=64,
                                                          shuffle=True)

    def calc_acc_and_loss(self, loader):
        self.model.eval()
        correct = 0
        total_images = 0
        total_loss = 0.0
        for images, labels in loader:
            predictions = self.model(images)
            _, predicted = torch.max(predictions, dim=1)

            total_loss += self.loss_fn(predictions, labels).item()

            total_images += labels.shape[0]
            correct += (predicted == labels).sum().item()
        return (correct / total_images), (total_loss / total_images)

    def calc_train_acc_and_loss(self):
        return self.calc_acc_and_loss(self.train_loader)

    def calc_val_acc_and_loss(self):
        return self.calc_acc_and_loss(self.val_loader)

    def print_info(self):
        if self.trained_epochs == 0:
            print("Model has not been trained")
        else:
            training_acc = self.training_acc_hist[len(self.training_acc_hist) - 1]
            training_loss = self.training_loss_hist[len(self.training_loss_hist) - 1]
            validation_acc = self.validation_acc_hist[len(self.validation_acc_hist) - 1]
            validation_loss = self.validation_loss_hist[len(self.validation_loss_hist) - 1]
            print(f"Epoch: {self.trained_epochs:3d}"
                  f" Training Acc: {training_acc:.3}"
                  f" Training Loss: {training_loss:.3}"
                  f" Validation Acc: {validation_acc:.3}"
                  f" Validation Loss: {validation_loss:.3}")

    def train(self, number_of_epochs, filepath=None, save=True):
        assert self.data is not None
        for epoch in range(number_of_epochs):
            self.trained_epochs += 1

            self.model.train()

            for images, labels in self.train_loader:
                predictions = self.model(images)

                loss = self.loss_fn(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            training_acc, training_loss = self.calc_train_acc_and_loss()
            validation_acc, validation_loss = self.calc_val_acc_and_loss()

            self.training_acc_hist.append(training_acc)
            self.training_loss_hist.append(training_loss)

            self.validation_acc_hist.append(validation_acc)
            self.validation_loss_hist.append(validation_loss)

            self.print_info()
        if save:
            assert filepath
            self.save_model(filepath)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    def plot_acc(self, title):
        plt.plot(self.training_acc_hist, label='training')
        plt.plot(self.validation_acc_hist, label='validation')
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def plot_loss(self, title):
        plt.plot(self.training_loss_hist, label='training')
        plt.plot(self.validation_loss_hist, label='validation')
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def overfit_single_batch(self, number_of_epochs):
        assert self.data is not None
        for epoch in range(number_of_epochs):
            self.trained_epochs += 1

            self.model.train()

            for images, labels in self.overfit_loader:
                predictions = self.model(images)

                loss = self.loss_fn(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            training_acc, training_loss = self.calc_acc_and_loss(self.overfit_loader)

            self.training_acc_hist.append(training_acc)
            self.training_loss_hist.append(training_loss)

            print(f"Epoch: {self.trained_epochs:3d}"
                  f" Training Acc: {training_acc:.3}"
                  f" Training Loss: {training_loss:.3}")

    def get_number_of_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prediction_from_url(self, url):
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        self.model.eval()
        _, prediction = torch.max(self.model(transform(img).unsqueeze(0)), 1)
        return self.classes[prediction.item()]
