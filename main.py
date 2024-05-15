import argparse
from dataset import load_data
from fed_client import Client
from fed_server import Server
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--epochs', type=int, default=Config.epochs, help='Number of training epochs')
    args = parser.parse_args()

    train_loader = load_data()
    server = Server()
    clients = [Client(train_loader) for _ in range(Config.num_clients)]

    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch + 1}/{args.epochs}")
        client_models = [client.train(server.broadcast()) for client in clients]
        server.aggregate(client_models)
        print(f"Epoch {epoch + 1} complete. Global model updated.")

if __name__ == "__main__":
    main()
 