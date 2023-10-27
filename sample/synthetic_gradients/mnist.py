import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class mnist():
    def __init__(self, args):

        train_dataset = dsets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
        
        test_dataset = dsets.MNIST(root='./data/',
                                    train=False,
                                    transform=transforms.ToTensor())
        
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,    
                                                        batch_size=args.batch_size,
                                                        shuffle=False)
        
        self.input_dims = 28*28
        self.num_classes = 10
        self.in_channel = 1
        self.num_train = len(train_dataset)