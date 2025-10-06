import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def anaphora_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming alignment of tokenization by token index
    for i in range(1, len_seq):
        token_text = tokenizer.decode(toks.input_ids[0][i])

        # Heuristic: attend to pronouns or connect repeated themes
        if token_text.lower() in ["her", "it", "they", "them", "she", "he", "you"]:
            for j in range(1, i):
                antecedent_text = tokenizer.decode(toks.input_ids[0][j])
                if antecedent_text.lower() not in ["her", "it", "they", "them", "she", "he", "you"]:
                    # Connect pronoun with an antecedent or theme
                    out[i, j] = 1
        if token_text.lower() in sentence.lower():  # Repeated themes
            for j in range(1, i):
                repeated_text = tokenizer.decode(toks.input_ids[0][j])
                if repeated_text.lower() in sentence.lower():
                    out[i, j] = 1

    # Handle cls and sep tokens
    out[0, 0] = 1  # CLS attends to itself
    out[-1, 0] = 1  # SEP attends to CLS

    # Normalize attention matrix row-wise
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Anaphora Resolution Pattern", out