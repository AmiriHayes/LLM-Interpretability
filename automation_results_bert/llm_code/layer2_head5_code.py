import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_end_importance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize the last token (usually punctuation or end of sentence) as an attention focal point
    end_idx = len_seq - 2  # Ignore the [SEP] token
    for i in range(1, end_idx + 1):  # Start after [CLS]
        out[i][end_idx] = 1  # Attention focused on sentence end token
        # Allow for uniform distribution
        if out[i].sum() == 0:
            out[i, -1] = 1.0  # Avoid having all zero rows by attending to the [SEP]

    out += 1e-4  # Small increment to remove divisions by zero
    out = out / out.sum(axis=1, keepdims=True)  # Row-wise normalization

    return "Sentence Final Marker Importance", out