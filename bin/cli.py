from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.train import train_model
from ser.transform import transform
from ser.data import Parameters
from bin.utils import EnhancedJSONEncoder

import typer
import json
import pickle
import os
from datetime import datetime

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "run"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        ..., "-b", "--batch-size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", "--learning-rate", help="The learning rate."
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #save the parameters!
    params = Parameters(lr=learning_rate, b=batch_size, e=epochs)
    # Best scores
    scores = {}

    # Create directory
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%d-%b-%Y_(%H:%M:%S)")
    directoryname = os.path.join(SAVE_DIR, name, f"{timestamp_str}")
    # Create directory if it does not exist
    Path(directoryname).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directoryname, "config.json"), "w") as fjson:
        json.dump(params, fjson, cls=EnhancedJSONEncoder)

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = transform()

    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    #Train
    train_model(epochs=epochs, training_dataloader=training_dataloader,
                validation_dataloader=validation_dataloader, model=model,
                optimizer=optimizer, device=device, directory=directoryname)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@main.command()
def infer():
    print("This is where the inference code will go")

if __name__ == "__main__":
    typer.run(train)