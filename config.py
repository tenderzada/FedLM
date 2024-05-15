import torch

class Config:
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5
    num_clients = 5
    epochs = 5
    use_cuda = True
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
