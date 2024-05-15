from model import SimpleCNN
from utils import federated_average
from config import Config


class Server:
    def __init__(self):
        self.global_model = SimpleCNN().to(Config.device)

    def aggregate(self, client_models):
        self.global_model = federated_average(client_models)

    def broadcast(self):
        return self.global_model
