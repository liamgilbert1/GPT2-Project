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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    '''
    Takes the raw token IDs all the way though the embeddings -> 12 blocks of attention + MLP -> final cleanup -> next token score predictions
    In the end, optionally computes how wrong those predictions were, if correct answers are given
    '''
    def forward(self, idx, targets=None):
        # idx is our batch of raw token ids, shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # looks up each token's meaning (wte) and each position's info (wpe), then adds them together to get our starting x
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # runs x through all n_layer blocks (attention + mlp), updating it a bit more each time
        for block in self.transformer.h:
            x = block(x)

        # final tidy up, then translate x into a score for every possible next token
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # if we were given the correct next tokens (training), figure out how wrong our guesses were
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    '''
    Utilizes our GPT2 model and populates it with trained GPT2 weights from Hugging Face
    '''
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # picks the size settings (n_layer, n_head, n_embd) based on which gpt2 version we want
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        
        # builds our own GPT class, currently just filled with random, untrained numbers
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # loads the real, already-trained gpt2 model from hugging face
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # makes sure our model's pieces line up with hugging face's before copying anything over
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # hugging face stores these specific weights sideways (an old openai format), so we need to flip them before copying
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # flips the sideways weights, then copies them into our model
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copies everything else over as-is, no flipping needed
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import tiktoken

'''
Hands out fresh (x, y) batches from a text file, one chunk at a time,
so the model isn't stuck training on the exact same tokens every step
'''
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # batch size
        self.T = T # sequence length

        # loads the whole file, tokenizes it once, and keeps it in memory for the rest of training
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches") # how many batches it takes to see every token once

        # tracks where we last left off reading, so the next call picks up fresh data
        self.current_position = 0

    def next_batch(self):
        # grabs the next chunk of tokens and builds our input (x) and target (y) - y is just x shifted over by one
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # moves our reading position forward, ready for the next call
        self.current_position += B * T

        # if there's not enough data left for another full batch, start back over from the beginning
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import time

# picks the fastest hardware available to run on
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# no multi-GPU (DDP) setup yet, so this is always the (only) master process
master_process = True

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


train_loader = DataLoaderLite(B=2, T=512)

torch.set_float32_matmul_precision('high')

# build a fresh, untrained model and run one forward pass to sanity check the output shape
model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, y)

# updates the models weights during training. takes the 'hint' from the gradients and nudges every number in the direction that reduces the error
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# repeats the guess, check, adjust cycle 50 times
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch() # grabs a fresh batch of training data
    x, y = x.to(device), y.to(device) # moves the data onto the same hardware as the model, so it can be processed
    optimizer.zero_grad() # clears out the old hints from last round, so they don't pile up
    logits, loss = model(x, y) # runs the forward pass, gets our predictions and how wrong they were
    loss.backward() # computes fresh hints (gradients) for every weight, based on this round's error
    optimizer.step() # nudges every weight using those hints
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    elif device == "mps":
        torch.mps.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")