import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(cnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_classes)

        self.cnn = nn.Sequential(
            self.layer1,
            self.layer2,
            self.fc
        )

        self.optimizers = []
        self.forwards = []
        self.init_optimizers()
        self.init_forwards()

    def init_optimizers(self, learning_rate=0.001):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_fc)
    
    def forward_layer1(self, x):
        out = self.layer1(x)
        return out
    
    def forward_layer2(self, x):
        out = self.layer2(x)
        return out
    
    def forward_fc(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out
    
    def forward(self, x):
        layer1  = self.layer1(x)
        layer2  = self.layer2(layer1)
        layer2_flat = layer2.reshape(layer2.size(0), -1)
        fc = self.fc(layer2_flat)
        return layer1, layer2, fc
