import utils
import torch
import torchvision
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from timeit import default_timer as timer

torchmetric_acc = Accuracy()

### Create Datasets
data_tr = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)
data_te = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)
print(len(data_tr), len(data_te)) # 60000, 10000 - black/white images of cloths

print(type(data_tr[0])) # tuple (tensor, int), tensor -> image, int -> index of label
print(data_tr[0][0].shape) # [color_channels, height, width]

# plot random image
rand_int = int(torch.randint(0, len(data_tr), size=[1]).item())
img, label_idx = data_tr[rand_int]
label = data_tr.classes[label_idx]
# utils.plot_img(img, label)

# divide data to batches by 32 items (60000/32 = 1875, 10000/32 = 313)
# so training loop will upload in memory only 32 images at a time, instead of 60k
BATCH_SIZE = 32
data_bch_tr = DataLoader(data_tr, BATCH_SIZE)
data_bch_te = DataLoader(data_te, BATCH_SIZE)
print(len(data_bch_tr), len(data_bch_te))

# get tensors from dataloader type
first_batch_images = torch.tensor([])
for step, x in enumerate(data_bch_tr):
    if step >= 1: break
    images, labels_idx = x
    first_batch_images = images
    label = data_tr.classes[labels_idx[0]]

### Model
class Model(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(), # important! we want to compress the image into a vector (height * width = 28 * 28 = 784)
            torch.nn.Linear(in_features=input_shape, out_features=hidden_units),
            torch.nn.Linear(in_features=hidden_units, out_features=output_shape))
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model = Model(28*28, 10, len(data_tr.classes))
state = model.state_dict()
# utils.print_state(state)

### Training Loop
torch.manual_seed(42)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 20
start_time = timer()
for epoch in range(epochs):
    train_loss = 0
    # loop trough the training batches
    for x, y in data_bch_tr:
        model.train()
        y_pred = model(x) # 1. forward pass
        loss = loss_fn(y_pred, y) # 2. calculate loss
        train_loss += loss
        optimizer.zero_grad() # 3. optimizer zero grad
        loss.backward() # 4. back propagation
        optimizer.step() # 5. optimizer step

    # calculate avg loss per batch
    train_loss /= len(data_bch_tr)

    # loop through the testing batches
    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_bch_te:
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y)
            test_accuracy += utils.accuracy_fn(y_pred.argmax(dim=1), y)

        # calculate avg loss per batch
        test_loss /= len(data_bch_te)
        # calculate avg accuracy per batch
        test_accuracy /= len(data_bch_te)

    print(f"epoch {epoch}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.4f}")

end_time = timer()
utils.print_time_diff(start_time, end_time)
