from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # makes sure all numbers can be evenly split among the heads
        # n_embd --> 3 * n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Used at the end of function forward to 'clean everything up'
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # saves the head and embd numbers for later reference
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # B = batch size (how many sequences being processed at once)
        # T = sequence length (how many tokens)
        # C = embedding size (n_embd)
        B, T, C = x.size()

        # runs x through translator converting n_embd to 3n_embd
        qkv = self.c_attn(x)

        # takes 3n_embd blob and splits it into 3 equal pieces amongst q, k and v
        # q - query, asks how similar it is to each key before it, including itself
        # k - key, used to be compared against every query
        # v - value, used to be blended together, weighted by how well q matched with k
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reorganizes q, k, and v from n_embd numbers per token, into n_head (6) separate groups
        # transposed so the math can be performed correctly, and swaps T (position 1) with n_head (position 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # for each word, dot product is performed to compare its query against every earlier words key (including its own)
        # the raw scores from the dot product get divided by a fixed number based on the head_size. Keeps scores from getting too large
        # is_causal=True guarantees a word can only be influenced by itself and earlier words, never future ones
        # the scaled scores then get converted into percentages that add up to 100%, and get blended together
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # undoes the transpose from earlier, glues all # of heads back together, side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # mixes the number of heads' separate results back together
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # expands the numbers to give the model more temporary workspace
        self.gelu    = nn.GELU(approximate='tanh') # math function that adds bendy behavior to the numbers (nonlinearity). Allows it to learn more complex, curvy patterns
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # shrinks the numbers back down to the original size
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # keeps values from getting too big or messy
        self.attn = CausalSelfAttention(config) # attention step - looks at the other words and gathers relevant info
        self.ln_2 = nn.LayerNorm(config.n_embd) # another tidy up step
        self.mlp = MLP(config) # processes what the attention step found (individually)

    '''
    Takes the word vectors in
    Adds what attention was learned from other words
    Adds what the MLP processed
    Returns the updated word vectors to pass onto the next block
    '''
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sentence length the model can recieve 
    vocab_size: int = 50257 # how many different tokens exists
    n_layer: int = 12 # how many repeated transformer blocks stacked on one another
    n_head: int = 12  # how many parrallel ways of seeing relationships between words (attention heads)
    n_embd: int = 768 # how many numbers describe each token's meaning

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        '''
        Assembles the models core parts
            wte: word lookup table
            wpe: position lookup table
            h: list of n_layer repeating transformer blocks
            ln_f: final cleanup step
        '''
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # translates the models n_embd value and turns it into the vocab_size value
        # One score per possible token in the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


