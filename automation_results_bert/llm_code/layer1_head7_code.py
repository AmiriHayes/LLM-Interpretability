from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokenized_sentence = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Assign high attention to punctuations and sentence boundaries
    for i, token in enumerate(tokenized_sentence):
        if token in {'.', ',', ';', ':', "?", "!", '[SEP]', '[CLS]'}:
            out[i, i] = 1  # Each punctuation attends to itself heavily

    # Ensure each row has attention summed to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to [SEP] if no specific attention
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Punctuation and Sentence Boundary Emphasis Pattern", out