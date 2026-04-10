import numpy as np
from transformers import PreTrainedTokenizerBase

def comma_focus_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The pattern we observed is that the sequence attends heavily to comma positions.
    for i, tok_id in enumerate(toks.input_ids[0]):
        # We identify commas in the token sequence.
        token = tokenizer.convert_ids_to_tokens(tok_id.item())
        if token == ',':  # Focus on comma tokens
            # Assign high attention scores to the commas and the tokens immediately preceding and following them
            if i > 0:
                out[i, i - 1] = 1  # Previous token
            if i + 1 < len_seq:
                out[i, i + 1] = 1  # Next token
            out[i, i] = 1  # Self-attention to comma itself

    # For rows with no focus, assign attention to the final token to avoid zero attention rows.
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention weights so they sum to 1 on each row
    out += 1e-4  # Prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Comma Focus Attention", out