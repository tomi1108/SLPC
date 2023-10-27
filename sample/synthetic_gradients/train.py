import torch
from model import *

class classifier():

    def __init__(self, args, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.batch_size = args.batch_size #100
        self.num_train = data.num_train #60000
        self.num_classes = data.num_classes #10

        self.net = cnn(data.in_channel, data.num_classes)

        if args.use_gpu:
            self.net.cuda()
        
        self.classificaationCriterion = nn.CrossEntropyLoss()
        self.num_epochs = args.num_epochs #300
        self.model_name = args.model_name
        self.best_perf = 0
        self.stats = dict(grad_loss = [], classify_loss = [])
        print(f"[{self.model_name}] model name will be")

    def train_model(self):
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader): #100データずつ実行
                labels_onehot = torch.zeros([labels.size(0), self.num_classes])
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                images = torch.tensor(images, requires_grad=True).cuda()
                labels = torch.tensor(labels).cuda()
                out = images

                for (optimizer, forward) in zip(self.net.optimizers, self.net.forwards):
                    out = self.optimizer_module(optimizer, forward, out, labels_onehot)
                
                loss = self.classificaationCriterion(out, labels)
                loss.backward()

                for (optimizer, forward) in zip(self.net.optimizers, self.net.forwards):
                    optimizer.step()

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                        %(epoch+1, self.num_epochs, i+1, self.num_train//self.batch_size, loss.item()))

    def optimizer_module(self, optimizer, forward, out, label_onehot=None):
        optimizer.zero_grad()
        out = forward(out)
        return out