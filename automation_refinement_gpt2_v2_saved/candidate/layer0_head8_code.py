from transformers import PreTrainedTokenizerBase
import numpy as np

def function_header_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Attempt to identify tokens associated with function definition
    for i, word in enumerate(words):
        if word.startswith('def'):
            out[i, i] = 1
            # Mark attention from function keyword to opening and closing parentheses, as well as function name
            j = i + 1
            while j < len(words) and (words[j] == '_' or words[j] == '(' or words[j] == ')'):
                out[i, j] = 1
                j += 1

    # Assign cls and eos self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention scores
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Function Header Pattern", out