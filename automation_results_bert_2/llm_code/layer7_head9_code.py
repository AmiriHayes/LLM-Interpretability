from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothesis Function

def predict_collocation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:

    # Initializing token sequence and attention matrix
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Splitting to evaluate based on tokens as input to the model
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    focus_words = {'share', 'found', 'sew', 'play', 'difficult', 'fix', 'thank', 'help', 'happy', 'feel', 'worked', 'smile', 'know'}

    # Identifying collocations by search in list of focused words and assigning attention internally within the phrase
    for i, word in enumerate(words):
        if any(focus_word in word for focus_word in focus_words):
            out[i, i + 1] = 1
            out[i + 1, i] = 1

    # Ensuring CLS and SEP self-attention (not a focus in our analysis, but implemented for compliance and model integration)
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize, prevent division by zero
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    # Outputting attention matrix
    return "Verb Phrase Collocation", out