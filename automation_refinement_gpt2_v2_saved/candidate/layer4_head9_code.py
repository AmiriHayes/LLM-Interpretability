import numpy as np
from transformers import PreTrainedTokenizerBase

def func_def_and_control_structures(sentence: str, tokenizer: PreTrainedTokenizerBase) -> [str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    attention_indices = []
    tokenized_sentence = toks.tokenizer.convert_ids_to_tokens(toks.input_ids[0].tolist())
    for idx, token in enumerate(tokenized_sentence):
        if token.startswith('def') or token == '(' or token == ')' or token == ':':
            attention_indices.append(idx)
        elif token.startswith('__') or token.isalnum():
            attention_indices.append(idx)

    for i in attention_indices:
        for j in range(len_seq):
            out[i, j] = 1

    out[0, 0] = 1  # self attention for cls
    out[-1, 0] = 1  # self attention for eos
    out += 1e-4
    return "Function Definition and Control Structure Pattern", out