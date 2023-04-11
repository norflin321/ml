import torch
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
steps = 5000
eval_interval = 500
learning_rate = 1e-3 # the self attention can't tolerate very very high learning rates
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
eval_batches_num = 200
n_embd = 32
#----------------

torch.manual_seed(1337)
torch.device(device)

with open("datasets/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

# here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
print(f"-- dataset has {vocab_size} unique characters:", itos)

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# this function averages up the loss over multiple batches,
# for each train and val sets, we generate many batches of data (how many specified by eval_batches_num variable),
# then measure a loss for each of them and get their average.
# so this will be a lot less noisy, then a loss inside training loop for a random batch
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_batches_num)
        for k in range(eval_batches_num):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(torch.nn.Module):
    """ one head of self attentions """
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # [B,T,C]
        q = self.query(x) # [B,T,C]
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # [B, T, C] @ [B, C, T] ---> [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # [B, T, T] 
        wei = F.softmax(wei, dim=-1) # [B, T, T]
        # perform the weighted aggregation of the values
        v = self.value(x) # [B,T,C]
        out = wei @ v # [B, T, T] @ [B, T, C] ---> [B, T, C]
        return out
    
class MultiHeadAttention(torch.nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
class FeedForward(torch.nn.Module):
    """ a simple linear layer followed by a non-linearity """

# BigramLanguageModel is a very simple model because the tokens are not talking to each other so given the previous context of
# what was generated we are only looking at the very last character to make a predictions about what comes next.
# To make better predictions for what comes next, tokens (characters) have to start talking to each other
# and figuring out what is in the context.
class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create embeding table which is a tensor of shape [64, n_embd] filled with random numbers
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # we cant just train a model on indices of characters from lookup table, before we need to embed them.
        # inputs shape is [batch_size, block_size] after embeding we should get a tensor
        # with shape [batch_size, block_size, 64], which means that each of block_size numbers in batch_size rows now represented
        # as an array of 64 random numbers. but why we need 64 numbers? Because we have so many unique characters in dataset
        # and each number indicates a probability for that number (but for now probabilities are random)
        tok_emb = self.token_embedding_table(idx) # [B,T,C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [T,C]
        # x holds not just the token ids but the positions at which these tokens occur
        x = tok_emb + pos_emb # [B,T,C]
        x = self.sa_heads(x) # apply one head of self-attention. [B, T, C]
        logits = self.lm_head(tok_emb) # [B,T,vocab_size]
        # now when we have predictions about what comes next, we would like to measure a loss (quality of predictions)
        # using cross_entropy (negative log likelihood), in order to do that we need to reshape logits and targets
        # (only because cross_entropy function expects data in different shapes)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_idx):
        # idx is [B, T] tensor of indices
        for _ in range(max_new_idx):
            # crop idx to the last block_size tokens
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_crop) # forward pass
            # focus only on the last time step
            logits = logits[:, -1, :] # logits now has shape of [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # [B, C]
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]
        return idx

model = BigramLanguageModel() # initializa the model
model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
for step in range(steps):
    # the loss inside training loop is a very noisy measurement of the current loss,
    # because every batch will be more or less lucky, so we will use estimate_loss function
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"--- step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    x, y = get_batch("train")
    # evaluate the loss
    logits, loss = model(x, y) # forward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate a sample
context = torch.zeros((1, 1), dtype=torch.long, device=device) # [[0]] - 0 maps to a new line character
print(decode(model.generate(context, max_new_idx=500)[0].tolist()))
