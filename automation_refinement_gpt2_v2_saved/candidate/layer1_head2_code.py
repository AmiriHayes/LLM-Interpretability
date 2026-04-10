import numpy as np
from transformers import PreTrainedTokenizerBase

def keyword_initialization_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a list of initialization keywords for functions and variable definitions
    init_keywords = {"def", "class", "import"}

    # Tokenize sentence to words using spaCy model
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Loop through tokens to find the main initialization keyword
    for i, word in enumerate(words):
        token = word.replace('Ġ', '') # Adjusting for tokenization specifics like GPT2
        if token in init_keywords:
            # Initialize the keyword to have higher self-attention
            for j in range(1, len_seq-1):
                out[j, i] = 1

    # Ensure CLS and EOS receive attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize output matrix
    out += 1e-4  # Add small constant to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize across rows

    return "Keyword Initialization Attention", out