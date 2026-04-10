import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Hypothesis: The head pays a particular attention to punctuation and sentence boundaries, 
# emphasizing periods, commas, and sentence start/end tokens by connecting them within the sentence.

def punctuation_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify sentence start, end tokens and punctuation marks
    start_tok_index = 0
    end_tok_index = len_seq - 1
    punctuation_indices = []

    # Determine punctuations and periods by using tokenizer specific IDs for punctuation
    punct_id = tokenizer.convert_tokens_to_ids(",")
    period_id = tokenizer.convert_tokens_to_ids(".")
    sentence_end_id = tokenizer.convert_tokens_to_ids("</s>")
    sentence_start_id = tokenizer.convert_tokens_to_ids("<s>")

    # Scan through the tokenized sentence
    for i, token_id in enumerate(toks.input_ids[0]):
        if token_id in [punct_id, period_id]:
            punctuation_indices.append(i)

    # Self-attention pattern for sentence start and end
    out[start_tok_index, start_tok_index] = 1
    out[end_tok_index, end_tok_index] = 1

    # Attention from punctuation to the sentence boundaries 
    for punct_idx in punctuation_indices:
        out[punct_idx, start_tok_index] = 1
        out[punct_idx, end_tok_index] = 1

    # Normalize the output matrix by rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Fill with attention to end token if no other attention present
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Sentence Boundary Detection", out