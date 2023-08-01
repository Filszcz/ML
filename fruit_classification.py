import torch
import random
import pandas as pd
import numpy as np
import os
import torchmetrics
import torch.utils.data
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms, datasets


# made as an exercise for the ML course
# https://www.learnpytorch.io/
# https://www.youtube.com/watch?v=Z_ikDlimN6A&t=73466s&ab_channel=DanielBourke
# https://www.youtube.com/watch?v=V_xro1bcAuA&ab_channel=freeCodeCamp.org

# data from kaggle
# https://www.kaggle.com/datasets/moltean/fruits


device = "cuda" if torch.cuda.is_available() else "cpu"
#complie the model later

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

train_path = Path("/home/filip/Documents/Code/Fruits/fruits-360_dataset/fruits-360/Training")
test_path = Path("/home/filip/Documents/Code/Fruits/fruits-360_dataset/fruits-360/Test")

data_transforms = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_path,
                                  transform=data_transforms,
                                  target_transform=None)


test_data = datasets.ImageFolder(root=test_path,
                                 transform=data_transforms)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)

classes = train_data.classes
#print(f"This dataset has {len(classes)} classes.")
#print(classes[0:5])

#tiny VGG


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # in_features = hidden units * pixels in the tensor after all the transformations
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #x = self.conv_block_1(x)
        #print(x.shape)
        #x = self.conv_block_2(x)
        #print(x.shape)
        #x = self.classifier(x)
        #print(x.shape)
        #return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=256, 
                  output_shape=len(train_data.classes)).to(device)

#summary(model_0, input_size=[1,3,64,64])

#torch.compile(model_0)

#train step 

EPOCHS = 20

""" optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.0001)

loss_fn = nn.CrossEntropyLoss() """

acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    
    model_0.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        #Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

model_0 = TinyVGG(input_shape=3, 
                  hidden_units=64,
                  output_shape=len(train_data.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.0001)

start_time = timer()

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=EPOCHS)

end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")

MODEL_PATH = Path("/home/filip/Documents/Code/Fruits/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Fruits_classification_model_0"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)



















