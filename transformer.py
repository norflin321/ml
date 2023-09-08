import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd

import os.path
from urllib.request import urlopen
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

print(f'pytorch version: {torch.__version__} device: {device}')

# hyperparameters
batch_size = 100 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
eval_interval = 500
learning_rate = 0.00035
eval_iters = 100
n_embd = 200
n_head = 5
n_layer = 5
dropout = 0.0
steps = 2500

def load_dataset():
    filename = 'datasets/tiny_shakespeare.txt'
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    if not os.path.isfile(filename):
        print('Downloading dataset...')
        data = urlopen(url).read()
        file = open(filename, 'wb')
        file.write(data)
        file.close()
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

text = load_dataset()
print(f'dataset len: {len(text)}')

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create tokenizer (encoding of a character to index)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8*len(data)) # first 90% will be train, rest val
train_data = data[:n]
test_data = data[n:]

print("train dataset:", train_data, len(train_data))
print("test dataset:", test_data, len(test_data))

# create data loader which generates a small batch of x (inputs) and corresponding y (targets)
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# one head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# a simple linear layer followed by a non-linearity
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

# transformer block (communication followed by computation)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# transformer model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :] # focus only on the last time step, -> (B, C)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, -> (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # # append sampled index to the running sequence, -> (B, T+1)
        return idx

# create model
model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f'model has {sum(p.numel() for p in model.parameters())} parameters')

# this function averages up the loss over multiple batches, for each train and val sets we generate
# many batches of data (how many specified by eval_batches_num variable), then measure a loss for
# each of them and get their average. so this will be a lot less noisy, then a loss inside training
# loop for a random batch
@torch.no_grad()
def estimate_loss(step):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            print('\r' + f'--- step {step}/{steps}, evaluate {split} {k}/{eval_iters} loss: {loss}', end='')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Karpathy best val loss 1.4873
# My best val loss 1.5512 (19min)
def train_model():
    print(f'{datetime.now().strftime("%H:%M:%S")} training start...')
    for step in range(steps):
        evaluated = False
        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0 or step == steps - 1:
            losses = estimate_loss(step)
            evaluated = True
            print('\r' + f'-- step {step}/{steps}, avg train loss {losses["train"]:.4f}, avg val loss {losses["val"]:.4f}')
        xb, yb = get_batch('train') # sample a batch of data
        logits, loss = model(xb, yb) # evaluate the loss
        if not evaluated:
            print('\r' + f'-- step {step}/{steps}, train loss {loss:.4f}', end='')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f'{datetime.now().strftime("%H:%M:%S")} training end...')

train_model()

# print('Generate from the model...')
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
