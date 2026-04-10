import numpy as np
from transformers import PreTrainedTokenizerBase
def clause_level_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence and map token positions
    encoded_input = tokenizer(sentence, return_offsets_mapping=True)
    offsets = encoded_input['offset_mapping']
    tokens = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Find positions of commas and conjunctions to infer clause boundaries
    comma_positions = [i for i, token in enumerate(tokens) if tokenizer.decode(token) == ',']
    conjunction_positions = [i for i, token in enumerate(tokens) if tokenizer.decode(token) in ['and', 'or', 'but']]

    # Consider both commas and conjunctions as clause separators
    clause_separators = sorted(set(comma_positions + conjunction_positions))

    # Mark attention within each clause
    current_start = 1  # Start after [CLS]
    for separator in clause_separators + [len_seq - 1]:  # Ensure to include the sequence end
        for i in range(current_start, separator + 1):
            for j in range(current_start, separator + 1):
                out[i, j] = 1
        current_start = separator + 1

    # Ensure no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Clause Level Attention", out