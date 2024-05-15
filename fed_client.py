import torch.optim as optim
from model import SimpleCNN
from config import Config
from copy import deepcopy
import torch.nn.functional as F


class Client:
    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.model = SimpleCNN().to(Config.device)

    def train(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(self.model.parameters(), lr=Config.learning_rate, momentum=Config.momentum)
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(Config.device), target.to(Config.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {batch_idx * len(data)}/{len(self.train_loader.dataset)} Loss: {loss.item():.6f}')
        return deepcopy(self.model)
