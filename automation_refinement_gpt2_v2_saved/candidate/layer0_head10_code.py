python
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def function_parsing_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initialize special token positions for attending
    out[0, 0] = 1  # CLS token
    out[-1, 0] = 1  # EOS attends to CLS

    # Function parsing pattern: focus on typical function construction keywords.
    keywords = ["def", "return", "import", "if", "for", "while"]

    # Create attention based on keywords
    for i, tok_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode(tok_id).strip()
        if token in keywords:
            out[i, i] = 1
            if i > 0:
                out[i, i-1] = 1  # Previous token attention

            # Also allow attention to previous keyword or opening parenthesis
            j = i - 1
            while j > 0:
                prev_token = tokenizer.decode(toks.input_ids[0][j]).strip()
                if prev_token in keywords or prev_token == "(":
                    out[i, j] = 1
                    break
                j -= 1

    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention
    return "Function Parsing Pattern", out