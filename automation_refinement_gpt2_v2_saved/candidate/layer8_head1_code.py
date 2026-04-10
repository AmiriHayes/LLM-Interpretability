from transformers import PreTrainedTokenizerBase
import numpy as np

def function_keywords_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize sentence and identify potential function name and keywords
    function_keywords = {'def', 'import', 'return', 'for', 'if', 'while'}
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    func_name_indices = [i for i, tok in enumerate(tokens) if tokens[i].lstrip('Ġ') in function_keywords]
    # Set self-attention for important tokens and attention to other key tokens
    for idx in func_name_indices:
        out[idx, idx] = 1
        for other_idx in func_name_indices:
            if idx != other_idx:
                out[idx, other_idx] = 1 / (len(func_name_indices) - 1 if len(func_name_indices) > 1 else 1)
    out[0, 0] = 1  # CLS-style self-attention
    out[-1, 0] = 1  # EOS connection
    # Normalize the final attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Function Name and Keyword Attention", out