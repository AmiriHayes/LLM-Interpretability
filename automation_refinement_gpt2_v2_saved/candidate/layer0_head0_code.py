import numpy as np
from transformers import PreTrainedTokenizerBase

def function_definition_anchor(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # This pattern anchors heavily to function definitions, especially at the 'def' keyword.
    for i in range(1, len_seq - 1):
        if "def" in tokenizer.convert_ids_to_tokens([toks.input_ids[0][i]])[0]:
            out[i, i] = 1  # The function word (def) attends to itself strongly.

    out[0, 0] = 1
    out[-1, 0] = 1
    return "Function Definition Anchor", out