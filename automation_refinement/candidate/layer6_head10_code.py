import numpy as np
from transformers import PreTrainedTokenizerBase

def comma_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    for i, token in enumerate(tokens):
        # Establish attention distribution on commas
        if token == ',':
            # Assign high attention to token after comma and before comma
            if i > 1:
                if tokens[i-1] == ",":
                    continue # avoid consecutive commas scenario
                out[i, i-1] = 1
            if i < len_seq - 2:  # ensuring there's a token after comma
                if tokens[i+1] == ",":
                    continue # avoid consecutive commas scenario
                out[i, i+1] = 1
        # Also, comma tokens have a reasonable 'spread' attention around sets
        elif tokens[i] in {"and", "but"}:
            out[i] = 0.5 if i-1 >= 0 and tokens[i-1] == "," else 0

    # Handle the special tokens attention
    out[0, 0] = 1  # CLS self-attention
    out[-1, 0] = 1  # SEP self-attention

    # Normalize attention matrix row-wise
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Comma Phrase Attention', out