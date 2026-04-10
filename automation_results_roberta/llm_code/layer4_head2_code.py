import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to determine significant words in a sequence based on a hypothesized role of Layer 4, Head 2

def significant_words_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> "Tuple[str, np.ndarray]":
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])  # Length of the tokenized sequences
    out = np.zeros((len_seq, len_seq))

    # Assign uniform attention to sentence boundaries
    out[0, 0] = 1.0  # [CLS] token attends to itself
    out[-1, -1] = 1.0  # [SEP] token attends to itself

    # We'll prioritize high attention to last tokens in each sentence segment to simulate importance
    # Example significant word indices (end of sentences or notable words may be calculated)
    # Since we have no clarity on the specific mechanism here's a simplified way
    for i in range(1, len_seq - 1):  # Skip [CLS] and [SEP]
        out[i, -1] = 1.0  # Let every token attend some weight to the final token as significant

    # Normalize to ensure a proper distribution of attention weights
    out += 1e-4  # Avoid division by zero if any rows are zero
    out /= out.sum(axis=1, keepdims=True)  # Normalize

    return "Most Significant Words Attention", out