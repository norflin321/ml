import torch
from torch.nn import functional as fun

with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(len(text))

# here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
print(encode("anton"))
print(decode(encode("anton")))

# let's now encode the entire text dataset and store it into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")


torch.manual_seed(1337)
class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets):
        # idx and targets are both (batch, time) tensor of integers
        logits = self.token_embedding_table(idx) # in shape of (batch, time, channel)
        b, t, c = logits.shape
        logits = logits.view(b*t, c)
        targets = targets.view(b*t)
        loss = fun.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
