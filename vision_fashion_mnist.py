import torch
import torchvision
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import utils

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
data_bch_tr = DataLoader(data_tr, 32)
data_bch_te = DataLoader(data_te, 32)
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
utils.print_state(state)
