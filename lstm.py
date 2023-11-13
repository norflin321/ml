import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(torch.__version__, device)

# @NOTE:
# It is LSTM multiclassification model (10 classes) trained on Fashion-MNIST dataset.
# Fashion-MNIST is a dataset comprising of 28×28 grayscale images.
# Training set has 60,000 images and the test set has 10,000 images.
# https://www.youtube.com/watch?v=J_ksCv_r_rU - explanation
# my best result: [avg test loss: 0.292882; test accuracy: 89.77%]

# @TODO:
# [x] try to move each tensor and layer to device? then try to only move model to device, is there a difference?
# [x] try to remove setting properties of LSTM class of hyperparameters and just use them from where they originally defined. does it matter?
# [.] master shapes of reshape() function
# [.] learn how to save and load model state_dict
# [.] without a tutorial figure out how to train this model on RGB images
# [.] implement the same model in tinygrad and compare ease of use and performance (https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)
# [.] implement transformer in the same way and compare results (https://github.com/norflin321/ml/blob/main/transformer.py)

# load Fashion-MNIST
train_data = datasets.FashionMNIST("./datasets", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())

# create Dataloaders with 100 images per batch
batch_size = 100
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# define hyperparameters
sequence_len = 28
input_len = 28
hidden_size = 128
n_layers = 2
n_classes = 10
n_steps = 10
lr = 0.01

class LSTM(nn.Module):
  def __init__(self):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_len, hidden_size, n_layers, batch_first=True, device=device)
    self.output_layer = nn.Linear(hidden_size, n_classes, device=device)
  def forward(self, input):
    hidden_states = torch.zeros(n_layers, input.size(0), hidden_size, device=device)
    cell_states = torch.zeros(n_layers, input.size(0), hidden_size, device=device)
    out, _ = self.lstm(input, (hidden_states, cell_states))
    out = self.output_layer(out[:, -1, :])
    return out

model = LSTM()
model = model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def train():
  t_start = time.time()
  n_batches = len(train_dataloader)
  print(f"Start training for {n_steps} steps on dataloader with {n_batches} batches of {batch_size} images in each batch...")
  for step in range(n_steps):
    total_loss = 0
    for images, labels in train_dataloader:
      images = images.reshape(-1, sequence_len, input_len)
      preds = model(images.to(device))
      loss = loss_fn(preds, labels.to(device))
      total_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    scheduler.step()
    print(f"-- step: {step+1}; loss: {(total_loss / n_batches):>4f}")
  print(f"trained in {int(time.time() - t_start)} sec")

train()

@torch.no_grad()
def test():
  n_batches = len(test_dataloader)
  total_loss, total_correct = 0, 0
  print(f"Start evaluating test dataset...")
  for images, labels in test_dataloader:
    images = images.reshape(-1, sequence_len, input_len)
    preds = model(images.to(device))
    loss = loss_fn(preds, labels.to(device))
    total_loss += loss.item()
    total_correct += (preds.argmax(1) == labels.to(device)).type(torch.float).sum().item()
  print(f"avg test loss: {(total_loss/n_batches):>4f}; test accuracy: {((total_correct/len(test_dataloader.dataset)) * 100):>0.2f}%")

test()
