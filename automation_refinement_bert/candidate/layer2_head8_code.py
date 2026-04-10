import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def question_relation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    question_punctuation_indices = [i for i, token in enumerate(decoded_tokens) if token in ['?', '.', ',']]

    attention_spans = []
    for idx in question_punctuation_indices:
        attention_span = 0
        if idx < len(decoded_tokens) - 1:
            for i in range(idx + 1, len(decoded_tokens)):
                if decoded_tokens[i] in ['?', '.', ',', ';', '[SEP]', '[CLS]']:
                    break
                attention_span += 1
        attention_spans.append((idx, attention_span))

    for idx, span in attention_spans:
        out[idx+1:idx+1+span, idx] = 1
        out[idx, idx+1:idx+1+span] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure no row is all zeros.
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Question Relation Attention", out