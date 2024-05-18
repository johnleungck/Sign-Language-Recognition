import numpy as np
import pandas as pd
import os

import torch
import torch.ao.quantization
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)

        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class QuantNet(nn.Module):
    def __init__(self):
        super(QuantNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 25)
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        x = self.dequant(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = Net().to(device)
checkpoint = r"weights/model.pt"
checkpoints = torch.load(checkpoint)

net_quant = QuantNet().to(device)
net_quant.load_state_dict(checkpoints)
net_quant.eval()

net_quant.qconfig = torch.ao.quantization.default_qconfig
net_quant = torch.ao.quantization.prepare(net_quant)

print(net_quant)

train_data = pd.read_csv(r"dataset/sign_mnist_train.csv")
test_data = pd.read_csv(r"dataset/sign_mnist_test.csv")

class SignsLanguageDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform

        if self.train == True:
            self.signs_lang_dataset = train_data
        else:
            self.signs_lang_dataset = test_data
        
        self.X_set = self.signs_lang_dataset.iloc[:, 1:].values
        self.y_set = self.signs_lang_dataset.iloc[:, 0].values

        self.X_set = np.reshape(self.X_set, (self.X_set.shape[0], 1, 28, 28)) / 255
        self.y_set = np.array(self.y_set)

    def __getitem__(self, index):
        image = self.X_set[index, :, :]
        label = self.y_set[index]
        sample = {'image_sign': image, 'label': label}
        return sample
    
    def __len__(self):
        return self.X_set.__len__()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['image_sign'].type(torch.FloatTensor).to(device)
            target = data['label'].type(torch.LongTensor).to(device)

            output = model(img)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

batch_size_test = 4
dataset_test = SignsLanguageDataset(train=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size_test)
test(net_quant, device, test_loader)

net_quant = torch.ao.quantization.convert(net_quant)
print(net_quant)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

print('Size of the model after quantization')
print_size_of_model(net_quant)

torch.save(net_quant.state_dict(), r"weights/QuantModel.pt")