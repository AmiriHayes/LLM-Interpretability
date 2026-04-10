import numpy as np
def quotation_association(sentence: str, tokenizer) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Initialize indices for QUOTATION
    quote_indices = [i for i, token in enumerate(tokens) if token in ["'", '"']]

    # Associate any pair of quotations
    for i in range(0, len(quote_indices), 2):
        if i+1 < len(quote_indices):
            out[quote_indices[i], quote_indices[i+1]] = 1
            out[quote_indices[i+1], quote_indices[i]] = 1

    # Make sure cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Quotation Association Pattern', out