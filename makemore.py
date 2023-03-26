import torch
import torch.nn.functional as f

words = open("datasets/names.txt", "r").read().splitlines()

## build the vocabulary of caracters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

## encode words in indexes from table and create training set of bigrams (x,y)
# xs = inputs, ys = targets
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    xs.append(stoi[ch1])
    ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

## we cant just feed indexes to nn, before we need to encode them into vectors
# this is how we can do it with one hot encoding and we want these numbers to be floats
xenc = f.one_hot(xs, num_classes=len(itos)).float()

## randomly initialize 27 neurons' weights. each neuron receives 27 inputs because
# we hot encoded each character into vector with length 27 (which has 26 zeroes and 1 unit)
g = torch.Generator().manual_seed(2147483647)
w = torch.randn((27, 27), generator=g, requires_grad=True)

## training loop
steps = 100
for k in range(steps):
    ## forward pass
    logits = xenc @ w # sign for matrix mul is "@"
    counts = logits.exp() # exponentiate the logits to get fake counts (idk wtf)
    probs = counts / counts.sum(1, keepdims=True) # normalize these counts to get probabilities
    # btw: the last 2 lines here are together called a "softmax" which is a normalization funtion,
    # you can put it on top of any other linear layer and it makes a nn to output probabilities.
    # now select the probabilities that the nn assigns to the correct next character
    loss = -probs[torch.arange(num), ys].log().mean() # and get their negative mean log likelihood (which is the loss)
    if (k % (steps/10) == 0):
        print("--", k, loss.item())
    ## backward pass
    w.grad = None # zero grad
    loss.backward()
    ## update
    w.data += -50 * w.grad # pyright: reportGeneralTypeIssues=false

## sample
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = f.one_hot(torch.tensor([ix]), num_classes=len(itos)).float()
        logits = xenc @ w
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
