from transformers import PreTrainedTokenizerBase
import numpy as np

def arithmetic_trends(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Patterns: Arithmetic operations and numbers
    operations = {'+', '-', '*', '/', '=', '^'}

    for i in range(len_seq):
        for j in range(len_seq):
            if i != j:
                if (i > 0) and ((toks.tokens()[i] in operations) or toks.tokens()[i].isdigit()):
                    if (toks.tokens()[j] in operations) or toks.tokens()[j].isdigit():
                        # If both tokens are part of arithmetic operations, they might relate
                        out[i, j] = 1

    # Adding small attention to [CLS] (index 0) and [SEP] (last index)
    out[0, 0] = 0.1
    out[-1, 0] = 0.1

    # Normalize output to form attention distribution
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Arithmetic Trends Detection", out