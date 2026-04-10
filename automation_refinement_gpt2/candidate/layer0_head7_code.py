from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Hypothesis: The head focuses on creating contextual groupings between tokens that are often divided by commas, except for conjunctions.

def comma_contextual_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using both space and transformer tokenizer
    words = sentence.split()
    comma_positions = [i for i, word in enumerate(words) if word.endswith(',')]

    # Map tokenized words to their respective positions
    token_to_idx = {i: toks.word_ids(batch_index=0)[i] for i in range(len(toks.input_ids[0]))}

    # Establish attention patterns that target contexts around commas without conjunctions
    for pos in comma_positions:
        if pos > 0 and pos + 1 < len(words):
            start_token = token_to_idx.get(min(pos, len(token_to_idx)-1))
            end_token = token_to_idx.get(min(pos+1, len(token_to_idx)-1))
            if start_token is not None and end_token is not None:
                # All tokens in the group before the comma should focus on the tokens after the comma
                for i in range(start_token):
                    out[i, end_token:end_token+2] = 1 / 2  # Focus distribution
                out[end_token:end_token+2, end_token:end_token+2] = 1 / 2  # Focus on both post-comma tokens

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default attention to the last token

    out += 1e-4  # Avoid division by zero for normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Comma Contextual Grouping", out

