import numpy as np
from transformers import PreTrainedTokenizerBase

def function_header_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    # Find positions of function header keyword 'def'
    def_indices = [i for i, token in enumerate(words) if token == 'def']
    # Attention pattern: enhance around 'def' keyword and follow function headers
    for def_index in def_indices:
        # Assuming function name always follows 'def'
        function_name_index = def_index + 1
        if function_name_index < len_seq:
            out[def_index + 1, function_name_index] = 1
            out[function_name_index, def_index + 1] = 1
    out[0, 0] = 1  # CLS token self-attention or beginning of sequence
    out[-1, 0] = 1  # attention from EOS token back to CLS or start of sequence
    out += 1e-4  # Slight noise to prevent zero rows
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
    return "Function Header Attention", out