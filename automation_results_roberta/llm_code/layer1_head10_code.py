import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_end_or_clause_end(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Predict self-attention for special tokens only
    out[0, 0] = 1.0  # <s> self-attention
    out[-1, -1] = 1.0  # </s> self-attention

    # Iterate over tokens, identify potential final or clause ending tokens
    for i in range(1, len_seq - 1):
        # Assign high attention if token corresponds to last or punctuation
        if toks.input_ids[0][i].item() in tokenizer.all_special_ids:
            out[i, -1] = 1  # End of sequence or clause
        # Checking for punctuation in `.`, `!`, `?`, `,`
        # To generalize for any punctuation, adjust logic with tokenizer specific ids
        elif toks.input_ids[0][i].item() in tokenizer.convert_tokens_to_ids(['.', '!', '?', ',']):
            out[i, -1] = 1 # Assigning high attention at possible sentence end positions

    # Avoid rows with all zeros - assign highest attention to last token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention scores across each row
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Final or Clause Ending Tokens Pattern", out
