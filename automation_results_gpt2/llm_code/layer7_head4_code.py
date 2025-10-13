import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def subject_pronoun_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    cls_index = 0
    eos_index = len_seq - 1

    # Define a list of potential subject pronouns
    subject_pronouns = {"I", "you", "he", "she", "it", "we", "they"}
    decoded_tokens = tokenizer.batch_decode(toks.input_ids[0], skip_special_tokens=False)

    # Without considering special tokens, get the attention pattern
    for i, token in enumerate(decoded_tokens):
        if token.strip() in subject_pronouns:
            for j in range(1, len_seq - 1):
                out[i, j] = 1 if i != j else 0 # Maximum focus on current pronoun, but zero self-attention

    # Assign some attention to cls and eos tokens
    for i in range(1, len_seq - 1):
        if out[i].sum() == 0:
            out[i, cls_index] = 1.0
        out[i, eos_index] = 1.0

    out = out / (out.sum(axis=1, keepdims=True) + 1e-4)
    return "Subject Pronoun Focus", out