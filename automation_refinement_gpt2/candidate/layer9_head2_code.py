import numpy as np
from transformers import PreTrainedTokenizerBase

def subject_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Typically, the first word or phrase in a sentence
    # acts as a subject or theme
    primary_token_index = 1

    # Assign high attention to the identified primary subject/theme
    for i in range(1, len_seq):
        out[i, primary_token_index] = 1

    # Set self-attention for [CLS] and [EOS] tokens if applicable in tokenizer
    out[0, 0] = 1  # [CLS] or start of sequence self-attention
    out[-1, -1] = 1  # [EOS] or end of sequence self-attention 

    # Normalize attention scores by row to account for divisibility
    out += 1e-4  # Avoids division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence-wide Emphasis on Primary Subject or Theme", out