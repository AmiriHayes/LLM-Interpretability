import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def verb_noun_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    token_dict = {i: tok for i, tok in enumerate(tokens)}

    # Patterns observed: verbs associated with nouns they might relate to
    for i, tok in token_dict.items():
        if tok in ["wanted", "went", "knew", "said", "smiled", "said", "felt"]:
            # find nearest noun according to our knowledge
            for j in range(i+1, len(tokens)):
                if tokens[j] in ["needle", "shirt", "button", "room", "lily", "mom"]:
                    out[i+1, j+1] = 1
                    out[j+1, i+1] = 1
                    break

    # Handle sequence start ([CLS]) and end ([SEP]) tokens with self-attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the out matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Verb-Noun Association Pattern", out