"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B" # folder we'll save all our shard files into
remote_name = "sample-10BT" # which version of the dataset to pull from hugging face (~10B tokens)
shard_size = int(1e8) # how many tokens go in each shard file (100M), so we end up with ~100 shards total

# creates the local folder to store our shards in, if it doesn't already exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# downloads the dataset from hugging face
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# sets up the same gpt2 tokenizer we've been using everywhere else
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # a special token used to mark where one document ends and the next begins
def tokenize(doc):
    # tokenizes one document, tagging the start with our end-of-text marker, and returns the token ids as a numpy array
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16) # shrinks the numbers down to a smaller data type, since token ids never get that big - saves disk space
    return tokens_np_uint16

# just saves a shard's worth of tokens to disk as a file
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenizes every document (in parallel, across multiple cpu cores) and writes them out into shard_size-sized shard files
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # a reusable buffer to build up the current shard's tokens in, before writing it to disk
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # if this document's tokens still fit in the current shard, just add them in
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # sets up the progress bar the first time, then updates it as tokens come in
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # the current shard is full - the very first shard becomes our val (held out) set, everything else is train
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # this document gets split across two shards - whatever fits goes in this one, the rest starts the next
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # starts the next shard off with this document's leftover tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # writes out whatever's left over as one final, smaller shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
