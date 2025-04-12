import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb833M"  # Changed directory for 833 million tokens
remote_name = "sample-10BT"
shard_size = int(1e8)  # Keep the shard size as it is

TARGET_TOKENS = 833 * 10**6  # 833 million tokens

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
# Load the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # End-of-text token

def tokenize(doc):
    """Tokenizes the document text into tokens."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """Writes tokenized data into a .npy file."""
    np.save(filename, tokens_np)

# Determine the number of processes to use (half of available CPU cores)
nprocs = max(1, os.cpu_count() // 2)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    total_tokens_collected = 0  # Track the total tokens collected
    progress_bar = None

    # Iterate through the dataset and tokenize the documents
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # Check if we've already collected enough tokens
        if total_tokens_collected + len(tokens) <= TARGET_TOKENS:
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                total_tokens_collected += len(tokens)

                # Initialize progress bar if it hasn't been set
                if progress_bar is None:
                    progress_bar = tqdm(total=TARGET_TOKENS, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write the full shard to file
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)

                # Prepare for the next shard
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        else:
            # Once the target tokens are collected, handle the last batch
            remaining_tokens_needed = TARGET_TOKENS - total_tokens_collected
            if remaining_tokens_needed > 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                all_tokens_np[:remaining_tokens_needed] = tokens[:remaining_tokens_needed]
                write_datafile(filename, all_tokens_np[:remaining_tokens_needed])
                total_tokens_collected += remaining_tokens_needed
            break

    # Final check to ensure we are not missing any tokens to be written
    if total_tokens_collected < TARGET_TOKENS:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

    # Close the progress bar if it was active
    if progress_bar:
        progress_bar.close()