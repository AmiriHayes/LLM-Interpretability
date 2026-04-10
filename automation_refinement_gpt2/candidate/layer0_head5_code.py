import numpy as np
from transformers import PreTrainedTokenizerBase

def numerical_critical_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokenized_words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    critical_tokens = {"number", "digits", "integer", "integers", "sum", "div", "multiple", "ask", "how", "total"}

    for i, token in enumerate(tokenized_words):
        # Detect tokens that are numbers (integer strings) or critical tokens and maximize their focus
        if token.isdigit() or any(word in token for word in critical_tokens):
            out[i, i] = 1.0

    # Add diagonal self-attention for start and end tokens
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Numerical and Critical Word Focus", out