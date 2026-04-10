import numpy as np
from transformers import PreTrainedTokenizerBase

def import_dependency_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> [str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    encoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify the positions of 'import' and the subsequent tokens
    import_indices = [i for i, token in enumerate(encoded_tokens) if token == 'import']

    for imp_index in import_indices:
        for i in range(imp_index, len_seq):
            out[imp_index][i] = 1
            if encoded_tokens[i] == "\n":
                break

    # Add attention to cls and eos tokens
    out[0, 0] = 1  # CLS
    out[-1, 0] = 1 # End of sentence

    # Normalize the output matrix
    out += 1e-4  # Prevent zero probability
    out = out / out.sum(axis=1, keepdims=True)

    return "Import Dependency Pattern", out