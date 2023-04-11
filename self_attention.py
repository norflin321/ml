import torch
import torch.nn.functional as F

# A mathematical trick that is used in the self attention inside a transformer
# and at the heart of an efficient implementation of self-attention

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print("x:", x.shape)

# Now we would like this 8 tokens in a batch to talk to each other. But the token for example at the fifth location
# should not communicate with future tokens in a sequence (6, 7, 8, ...). It should only talk to tokens in (4, 3, 2, ...) locations.
# So information only flows from previous context to the current timestamp and we cannot get any information from the future
# because we are about to try to predict the future.
# The easiest way for tokens to communicate is to just do an average of all the preceding elements.
# For example if i am the fifth token (T) i would like to take channels (C) that make up information at my step but
# then also the channels from the four, third, second and first steps. I'd like to average those up and then that would
# become sort of like a feature vector that summarizes me in the context of my history.

# for each T inside each B, we wanna calculate the average of current T and all the previous
xbow = torch.zeros((B,T,C))

def version_01():
    for b in range(B):
      print("---- b:", b)
      for t in range(T):
        cur_and_prev = x[b, :t+1]
        mean = torch.mean(cur_and_prev, 0)
        xbow[b,t] = mean
        print(f"-- t: {t}, cur_and_prev: {cur_and_prev.shape}, mean: {mean}")

    print(xbow[0])
    print(x[0])

def version_02():
    # version 2 (the same result) using tril, softmax and matrix mul.
    tril = torch.tril(torch.ones(T,T))
    wei = torch.zeros((T,T))
    print(wei)
    wei = wei.masked_fill(tril == 0, float("-inf"))
    print(wei)
    wei = F.softmax(wei, dim=-1) # normalization to 1
    print(wei)
    xbow2 = wei @ x
    print(torch.allclose(xbow, xbow2))

def version_03():
    # version 3: self-attention!

    # Different tokens will find different other tokens more or less interesting so each token wants to gather
    # information from the past in a data depended way, and this is the problem that self-attention solves.
    # The way self-attention solves it:
    # Every single token at each position will emit two vectors: query and key.
    # The query vector is "what am i looking for?"
    # The key vector is "what do i contain?"
    # So the way we get affinities (родство) between these tokens in a sequence: we do a dot product between the keys and the queries.
    # So my query dot products with all the keys of all the other tokens and that dot product becomes "wei" (variable in example below).
    # So if the key and the query are sort of aligned they will interact to a very high amount and then i will get to learn more about
    # that specific token as opposed to any other token in a sequence.

    torch.manual_seed(1337)
    B,T,C = 4,8,32 # batch, time, channels
    x = torch.randn(B,T,C)

    # let's see a single Head perform self-attention
    head_size = 16

    # these layers just going to apply a matrix mul with some fixes waights (because bias=False)
    key = torch.nn.Linear(C, head_size, bias=False)
    query = torch.nn.Linear(C, head_size, bias=False)
    value = torch.nn.Linear(C, head_size, bias=False)

    k = key(x) # [B, T, 16]
    q = query(x) # [B, T, 16]

    wei = q @ k.transpose(-2, -1) # [B, T, 16] @ [B, 16, T] ---> [B, T, T]
    print("-- outputs of the dot product is raw affinities between all tokens:")
    print(wei[0], "\n")

    tril = torch.tril(torch.ones(T,T))
    wei = wei.masked_fill(tril == 0, float("-inf")) # just delete this line if you want to allow all tokens talk to each other fully (ex sentimate analysis)
    print("-- but we want to ban some tokens from communication (explanation why see at the top of the notebook)")
    print(wei[0], "\n")
    wei = F.softmax(wei, dim=-1) # normalization
    print("-- now we use softmax (exponentiate and normalized) because we want to have a nice distribution that sums to one")
    print(wei[0])

    v = value(x)
    out = wei @ v
    out.shape

    # Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating
    # information with a weighted sum from all nodes that point to them, with data-dependent weights.

    # There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.

    # Each example across batch dimension is of course processed completely independently and never "talk" to each other.

    # In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate.
    # This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.

    # "self-attention" just means that the keys and values are produced from the same source as queries.
    # In "cross-attention", the queries still get produced from `x`, but the keys and values come from some other, external source (e.g. an encoder module).

version_03();
