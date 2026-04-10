from transformers import PreTrainedTokenizerBase
import numpy as np

def function_definition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Detect tokens that are part of function definitions.
    word_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    for i, token in enumerate(word_tokens):
        # Looking for 'def' keyword and following tokens as function elements
        if token == "Ġdef":  # GPT-2 byte-level BPE token for ' def' (intending leading space)
            # Attend to the function name and parameters elements following 'def'
            j = i + 1
            # Allow up to 4 tokens after 'def' to be attended strongly
            while j < i + 5 and j < len(word_tokens) and (word_tokens[j].isalnum() or word_tokens[j] == '_'):
                out[i, j] = 1 
                out[j, i] = 1
                j += 1

    # Set self-attention weights
    np.fill_diagonal(out, 1)

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Pattern", out