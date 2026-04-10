from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def direct_indirect_speech_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize and extract token id mappings
    token_ids = toks.input_ids.tolist()[0]
    token_map = {i: tokenizer.decode([token]).strip() for i, token in enumerate(token_ids)}

    # Define patterns of interest
    quote_indices = [i for i, token in token_map.items() if token in {'"', "'"}]
    comma_indices = [i for i, token in token_map.items() if token == ',']
    question_indices = [i for i, token in token_map.items() if token == '?']

    # Ensure quotes have pairs
    if len(quote_indices) % 2 != 0:
        quote_indices = quote_indices[:-1]

    # Process quotes - targeting speech within quotes
    for start_idx, end_idx in zip(quote_indices[::2], quote_indices[1::2]):
        out[start_idx+1:end_idx, start_idx+1:end_idx] = 1
        out[start_idx:end_idx, start_idx] = 0.1

    # Process common separator tokens (commas followed by speech) and direct questions
    for idx in comma_indices + question_indices:
        segment_start = idx
        while segment_start > 0 and out[segment_start].sum() == 0:
            segment_start -= 1
        out[segment_start+1:, idx] = 0.5

    # Normalize outputs to have valid distribution
    out += np.eye(len_seq) * 1e-4  # Avoid zero rows with self attention
    out = out / out.sum(axis=1, keepdims=True)
    return "Direct and Indirect Speech Attention Pattern", out