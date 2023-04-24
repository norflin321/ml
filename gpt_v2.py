# https://youtu.be/d7IRM40VMYM

import torch
import torch.nn as nn

from transformers.activations import gelu_new

class CustomGELU(nn.Module):
    """ GELU implementation taken from the `transformers` """
    def forward(self, x):
        """ Run forward pass """
        return gelu_new(x)

class Block(nn.Module):
    """ Decoder block
    Parameters:
        n_embd: (int) Dimensionality of the embeddings
        n_head: (int) Number of attention heads
        n_positions: (int) Maximum number of tokens
        attn_pdrop: (float) Probability of dropout on attention weights
        resid_pdrop: (float) Probability of dropout after applying the MLP
        layer_norm_epsilon: (float) Hyperparameter of layer normalization
    Attributes:
        ln_1, ln2: (nn.LayerNorm) Layer norms.
        attention: (nn.MultiHeadAttention) Attention module.
        mlp: (nn.Sequential) Multilayer perceptron.
    """
    def __init__(self, n_embd, n_head, n_positions, attn_pdrop, resid_pdrop, layer_norm_epsilon):
        super().__init__()
        # Here we instantiate our two layer normalization modules.
        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        # Here we instantiate the PyTorch multi-head attention module and we feed in all the relevant hyper parameters.
        # Repo of Andrej Karpathy contain the manual implementation, but here it is outsourced to PyTorch.
        self.attention = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, bias=True, batch_first=True)
        # instantiate 'no lookahad mask' - it is a lower triangular matrix and the elements that are equal to True
        # will be those that are not going to be considered for the self-attention.
        self.register_buffer('mask', (1 - torch.tril(torch.ones(n_positions, n_positions))).to(dtype=torch.bool))
        # finally we prepare the multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop)
        )
    def forward(self, x):
        """ Run forward pass. Here you can see the sketch of the forward pass: https://i.imgur.com/qT3uwZY.png
        Parameters:
            x: (torch.Tensor) Input tensor of shape `(batch_size, n_tokens, n_embd)`.
        Returns:
            (torch.Tensor) Output tensor of shape `(batch_size, n_tokens, n_embd)`.
        """
        # first of all, we get the actual values of all three of the dimensions.
        batch_size, n_tokens, n_embd = x.shape
        # then we start working on the first residual block which starts with a layer normaliztion.
        x_ = self.ln_1(x) # (batch_size, n_tokens, n_embd)
        # we dynamically cut off our attention mask buffer, because the registered buffer
        # has the maximum size of number of positions times number of positions
        mask = self.mask[:n_tokens, :n_tokens] # (n_tokens, n_tokens)
        # here we run forward pass of the attention module. we basically provide 3 times our layer normalized
        # tensor, that will serve as the key, value and query. We also provide the prepared mask. And finally
        # we tell torch that we don't need to have the attention weights, since we are not really going to use them,
        # however they are actually useful when it comes to trying to explain what the model wa looking at and so on.
        attn_out, _ = self.attention(x_, x_, x_, attn_mask=mask, need_weights=False) # (batch_size, n_tokens, n_embd)
        # here we basically take the output of the residual black and add it to the original tensor.
        x = x + attn_out # (batch_size, n_tokens, n_embd)
        # and here we just implement the second residual block in one line. We applied layer normalization and
        # then we ran the tensor through the multi-layer perceptron and again added it to the input tensor.
        x = x + self.mlp(self.ln_2(x)) # (Batch_size, n_toekns, n_embd)
        return x

class GPT(nn.Module):
    """ Entire GPT model.
    Parameters:
        vocab_size: (int) Number of tokens in the vocamulary. We need to to be able to define the token embedding table.
        n_layer: (int) Number of decoder blocks to include.
        n_embd: (int) Dimensionality of the embeddings.
        n_head: (int) Number of attention heads.
        n_positions: (int) Maximum number of tokens.
        attn_pdrop: (float) Probability of dropout on attention weights.
        embd_pdrop: (float) Probability of dropout on the sum of embeddings.
        resid_pdrop: (float) Probability of dropout after applying the MLP.
        layer_norm_epsilon: (float) Hyperparameter of layer normalization.
    Attributes:
        token_emb: (nn.Embedding) Token embeddings.
        pos_emb: (nn.Embedding) Positional embedding.
        drop: (nn.Dropout) Dropout module to be applied on the sum of embeddings.
        blocks: (nn.Sequential) List of decoder blocks.
        ln: (nn.LayerNorm) Layer of normalization that applied before applying 'head'.
        head: (nn.Linear) Final linear layer, that maps from the hidden embedding space to vocabulary space.
    """
    def __init__(self, vocab_size, n_layer, n_embd, n_head, n_positions, attn_pdrop, embd_pdrop, resid_pdrop, layer_norm_epsilon):
        super().__init__()
        self.n_positions = n_positions
        # Here we define our two embedding modules. The first one is the token embeddings and basically for each
        # token in our vocabulary we will have an embedding vector. And also we create the positional embeddings.
        # The reason why we need these positional embeddings is that we want to encode some information about the
        # position of a given token and let's say a given sentence.
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(vocab_size, n_embd)
        # Here we define a dropout module.
        self.drop = nn.Dropout(embd_pdrop)
        # Here we define a sequential module which will be a list of decoder blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop, layer_norm_epsilon) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, idx):
        """ Run Forward pass. It will return logits for all tokens and not just the last one.
        And also we are not goint to apply softmax inside of this forward method.
        Because we will do both of these operations later, when we implement sampling.
        Here is the diagram of the first part of the forward pass: https://i.imgur.com/YmS6WyJ.png
        Here is the diagram of the secod part: https://i.imgur.com/s8f9r0F.png
        Note that we are going to end the forward pass after the linear module, and we will extract the last token
        and apply softmax in a different function.
        Parameters:
            idx: (torch.Tensor) Integer tensor of shape (batch_size, n_tokens) where each item is in the range of [0, vocab_size]
        Returns:
            logits: (torch.Tensor) Tensor of shape (batch_size, n_tokens, vocab_size)
        """
        batch_size, n_tokens = idx.shape
        device = idx.device
        # Here we just make sure that if the user tries to provide too many tokens then we raise and exception.
        if n_tokens > self.n_positions:
            raise ValueError('There are too many tokens in the input')
        # Here we just create a tensor that represents the positions.
        positions = torch.arange(n_tokens, device=device) # (n_tokens)
        # Here we take our input token indices and we get their corresponding embeddings.
        token_emb = self.token_emb(idx) # (batch_size, n_tokens, n_embd)
        # then we take the positions and get their positional embeddings and prepend it with an extra dimension,
        # because we actually want to element-wise sum it with token embeddings and we want torch to do broadcasting
        # over the batch dimension.
        pos_emb = self.pos_emb(positions)[None, ...] # (1, n_tokens, n_embd)
        # Here we sum the two embeddings and we apply a dropout.
        x = self.drop(token_emb + pos_emb) # (batch_size, n_tokens, n_embd)
        # Here we run out tensor through each of the decoder blocks we have.
        x = self.blocks(x) # (batch_size, n_tokens, n_embd)
        x = self.ln(x) # (batch_size, n_tokens, n_embd)
        # and finally we get our logits over all tokens in the vocabulary by applying a linear module.
        logits = self.head(x) # (batch_size, n_tokens, vocab_size)
        return logits

@torch.no_grad()
def generate_token(model, token_ixs, temperature=1.0, sample=False, top_k=None):
    """ Generate a single token given previous tokens.
    Parameters:
        model: (GPT) Out GPT model instance.
        token_ixs: (list) List of token of the input text.
        temperature: (float) The higher the more variability and vice verse.
        sample: (bool) If True, we sample from the distribution. If False we just take the argmax
        top_k: (int or None) If not None then we modify the distribution to only contain the 'top_k' most probable outcomes.
    Returns:
        new_token_ix: (int) Index of the new token.
    """
    # Here we make sure that if the user provided too many tokens then we only take most recent ones.
    context_token_ixs = token_ixs[-model.n_positions:]
    # there we take the list of tokens and cast it to a torch tensor and we also prepend it with a dummy batch dimension.
    ixs = torch.tensor(context_token_ixs).to(dtype=torch.long)[None, :] # (1, n_tokens)
    # here we run the forward pass and get the logits.
    logits_all = model(ixs) # (1, n_tokens, vocab_size)
    # here we take only the first sample because our batch was composed only of the first sample, and also
    # we only take the embedding corresponding to the very last token and throw away all the remaining token embeddings.
    logits = logits_all[0, -1, :] # (vocab_size)
    # here we just divide by the temperature
    logits = logits / temperature # (vocab_size)
    # find the top k biggest elements inside the logits tensor, set the remaining elements to -inf,
    # we set them to -inf because after that we will take this logits tensor and run it through the softmax activation,
    # and after that all the -inf logits will actually end up having exactly zero probability.
    if top_k is not None:
        top_values, _ = torch.topk(logits, top_k) # (top_k)
        logits[logits < top_values.min()] = -torch.inf
    # this is where we create the probabilities
    probs = torch.nn.functional.softmax(logits, dim=0) # (vocab_size)
    # then if we want to have randomness, we sample and internally this is done through the multinomial distribution in torch,
    # however if we don't want to sample and don't want to have any randomness we just take the argmax which would
    # correspond to the token with the highest probability.
    if sample:
        new_token_ix = torch.multinomial(probs, num_samples=1)
    else:
        new_token_ix = probs.argmax()
    return new_token_ix.item()
