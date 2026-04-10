from transformers import PreTrainedTokenizerBase
import numpy as np

# Function that identifies coordinating conjunctions and links them
# to the preceding and succeeding clauses.
def coordinating_conjunction_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing sentence into words (aligned with tokenized indices)
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # A simple list of coordinating conjunctions
    conjunctions = {"and", "but", "or", "yet", "for", "nor", "so"}

    # Identify coordinating conjunctions and construct the matrix pattern
    for i in range(1, len_seq - 1):
        if words[i] in conjunctions:
            out[i, i] = 1
            if i > 1:  # Link to the previous token (before conjunction)
                out[i, i - 1] = 1
            if i < len_seq - 2:  # Link to the next token (after conjunction)
                out[i, i + 1] = 1

    # CLS and SEP token self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize out matrix by row for consistent attention
    out = out / out.sum(axis=1, keepdims=True, where=out.sum(axis=1) != 0)

    return "Coordinating Conjunction Linkage", out