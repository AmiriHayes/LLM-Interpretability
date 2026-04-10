import numpy as np
from transformers import PreTrainedTokenizerBase
def initial_token_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Create attention to the first non-special token
    first_non_special_token_index = 1 if len_seq > 2 else 0  # Handle special cases and only single words
    for i in range(1, len_seq - 1):
        out[i, first_non_special_token_index] = 1  # Attend to the first word
    out[0, 0] = 1  # Self-attention for CLS token
    out[-1, 0] = 1  # Attend from EOS to first token
    # Normalize the output matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Alignment", out