from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# The main function to predict the attention pattern based on the hypothesis
# of handling document structure and boundaries.
def document_structure_handler(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus primarily on the start and end of the sentence, as well as sentence separators
    # which typically would mean the <s>, </s> tokens or similar.
    sentence_tokens = toks.input_ids[0].tolist()

    start_token_index = 0
    end_token_index = len_seq - 1

    # Assign dominant attention at sequence boundaries with <s> and </s>
    out[start_token_index, start_token_index] = 1  # Self-attention at beginning
    out[start_token_index, end_token_index] = 1    # Pay attention to end (closing the sentence)
    for i in range(1, len_seq - 1):
        out[i, start_token_index] = 0.8  # Pay strong attention to start of the sentence
        out[i, end_token_index] = 0.2    # Pay lesser attention to the end/conclusion

    # Pass structure through sentence closing token self-attention
    out[end_token_index, start_token_index] = 1  # End pays attention back to start
    out[end_token_index, end_token_index] = 1    # Self-attention at the end

    # Ensure no row has zero weights
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention weights
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize the matrix across each row

    return "Document Structure and Sentence Boundary Handler", out