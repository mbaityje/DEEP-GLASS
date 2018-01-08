import torch.nn.functional as F
import torch.nn as nn
import torch


torch.set_default_tensor_type('torch.DoubleTensor')

def loadNet(pre_path,model):
    args = torch.load(pre_path)
    model.load_state_dict(args)
    return model

class convAlexrelu(nn.Module):
    def __init__(self, layers=[3,64,64,192,10], k_dims=[5,5],im_dim = 32):
        super(convAlexrelu, self).__init__()
        self.conv1 = nn.Conv2d(layers[0], layers[1], kernel_size=k_dims[0],stride=2)
        self.conv2 = nn.Conv2d(layers[1], layers[2], kernel_size=k_dims[1],stride=2)
        self.view_size = im_dim
        for k in k_dims:
            self.view_size -= k-1
            self.view_size /= 2
            self.view_size -= 2
        self.view_size = self.view_size*self.view_size*layers[2]
        self.fc1 = nn.Linear(self.view_size, layers[3])
        self.fc2 = nn.Linear(layers[3], layers[4])
        self.nonlins = {'conv1':('max_relu',(2,2)),'conv2':('max_relu',(2,2)),'fc1':'relu','fc2':'log_softmax'}

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3,stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3,stride=1))
        x = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class AlexNet(nn.Module):

    def __init__(self, layers=[3,64,128,512,128,10],raw=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(layers[0], layers[1], kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(layers[1], layers[2], kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(layers[2] * 4 * 4, layers[3])
        self.fc2 = nn.Linear(layers[3], layers[4])
        self.fc3 = nn.Linear(layers[4], layers[5])
        self.layers = layers
        if raw:
            self.features = nn.Sequential(
                self.conv1,
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                self.conv2,
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                self.fc1,
                nn.ReLU(inplace=True),
                self.fc2,
                nn.ReLU(inplace=True),
                self.fc3
        )
        else:
            self.features = nn.Sequential(
                self.conv1,
                nn.BatchNorm2d(layers[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1),
                self.conv2,
                nn.BatchNorm2d(layers[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1),
            )
            self.classifier = nn.Sequential(
                self.fc1,
                nn.BatchNorm1d(layers[3]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                self.fc2,
                nn.BatchNorm1d(layers[4]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                self.fc3
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.layers[2] * 4 * 4)
        x = self.classifier(x)
        return F.log_softmax(x)


class conv22relu(nn.Module):
    def __init__(self, layers=[1,5,10,80,10], k_dims=[5,5],im_dim = 28):
        super(conv22relu, self).__init__()
        self.conv1 = nn.Conv2d(layers[0], layers[1], kernel_size=k_dims[0])
        self.conv2 = nn.Conv2d(layers[1], layers[2], kernel_size=k_dims[1])
        self.view_size = im_dim
        for k in k_dims:
            self.view_size -= k-1
            self.view_size /= 2
        self.view_size = int(self.view_size*self.view_size*layers[2])
        self.fc1 = nn.Linear(self.view_size, layers[3])
        self.fc2 = nn.Linear(layers[3], layers[4])
        self.nonlins = {'conv1':('max_relu',(2,2)),'conv2':('max_relu',(2,2)),'fc1':'relu','fc2':'log_softmax'}

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class conv22tanh(nn.Module):
    def __init__(self, layers=[1,5,10,80,10], k_dims=[5,5],im_dim = 28):
        super(conv22tanh, self).__init__()
        self.conv1 = nn.Conv2d(layers[0], layers[1], kernel_size=k_dims[0])
        self.conv2 = nn.Conv2d(layers[1], layers[2], kernel_size=k_dims[1])
        self.view_size = im_dim
        for k in k_dims:
            self.view_size -= k-1
            self.view_size /= 2
        self.view_size = self.view_size*self.view_size*layers[2]
        self.fc1 = nn.Linear(self.view_size, layers[3])
        self.fc2 = nn.Linear(layers[3], layers[4])
        self.nonlins = {'conv1':('max_tanh',(2,2)),'conv2':('max_tanh',(2,2)),'fc1':'tanh','fc2':'log_softmax'}

    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.view_size)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class singleHiddenFullyConnected(nn.Module):
    def __init__(self, layers=[3072,10,10]):
        super(singleHiddenFullyConnected, self).__init__()
        self.layers=layers
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x.view(-1,self.layers[0]))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out




