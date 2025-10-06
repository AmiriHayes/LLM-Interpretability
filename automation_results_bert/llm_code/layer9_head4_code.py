import numpy as np
from transformers import PreTrainedTokenizerBase

def quotation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify positions of quotation marks
    quote_indices = [i for i, word in enumerate(words) if word == '"']

    for index in quote_indices:
        # Each quotation mark attends to each other quotation mark strongly
        for other_index in quote_indices:
            if index != other_index:
                out[index, other_index] = 1

    # Normalize so that each row sums to 1 (except for [CLS] and [SEP])
    out[:1, :] = 1 / (len_seq - 1)  # [CLS] attention distribution
    out[-1:, :] = 1 / (len_seq - 1)  # [SEP] attention distribution

    for row in range(1, len_seq-1):
        if words[row] == '"':
            out[row, :] /= out[row, :].sum()  # Normalize row
        else:
            out[row, -1] = 1.0  # Default to attending [SEP] if no specific attention

    return "Direct Speech and Quotation Pattern", out