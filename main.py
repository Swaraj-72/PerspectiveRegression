import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.CNNRegressor.model import CNNRegression
from src.Dataset.dataset import CustomDataset

# check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def main():
    # prepare data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(csv_file='artifacts/car_imgs_4000.csv', 
                                                transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = (len(dataset) - train_size) // 2
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                        [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # prepare model
    model = CNNRegression()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train the model
    for epoch in range(15):  # Adjust number of epochs as needed
        print(f"training epoch: {epoch + 1}")
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            print("batch loaded")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}")
    # evaluate the model on validation and test sets
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    test_loss_final = test_loss / len(test_loader
    print(f"Test Loss: {test_loss_final}")

    if test_loss_final < 0.01:
        torch.save(model.state_dict(), f'model/model_{test_loss_final}.pt')
        

if __name__ == "__main__":
    main()

