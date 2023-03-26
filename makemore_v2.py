import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt

words = open("datasets/names.txt", "r").read().splitlines()

## build the vocabulary of caracters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

## build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?
x, y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w + ".":
        idx = stoi[ch]
        x.append(context)
        y.append(idx)
        print("".join(itos[i] for i in context), "-->", itos[idx])
        context = context[1:] + [idx] # crop and append

x = torch.tensor(x) # shape: [32, 3]
y = torch.tensor(y) # shape: [32]

## create a lookup table with random numbers, it will have 27 rows and 2 columns.
# so each one of 27 characters will have a two-dimensional embedding
c = torch.randn((27, 2))

## so after embeding each item of "x" for example first [0, 0, 0] will be [c[0], c[0], c[0]]
# but c[0] is an array of 2 random numbers, so each character of context will be
# represented as an array of two random numbers
emb = c[x]
print("emb:", emb.shape) # [32, 3, 2]

## create random weights and biases for first layer
w1 = torch.randn((6, 100))
b1 = torch.randn(100)

## resize "emb" into [32, 6] shape and matrix mul with first layer (forward pass)
# then pass it to 10h actvation function, witch makes each number are number between -1 and 1
h = torch.tanh(emb.view(emb.shape[0], 6) @ w1 + b1)
print("first layer + 10h:", h.shape) # [32, 100]

## create final layer
w2 = torch.randn((100, 27))
b2 = torch.randn(27)

## forward pass to final layer to get nn output
logits = h @ w2 + b2

## get probabilities
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
print("prob:", prob.shape) # [32, 27]
print("every row of prob is normalized:", prob[0].sum()) # 1.0

## check how right probs are and get loss
loss = -prob[torch.arange(32), y].log().mean()
print("loss:", loss)
