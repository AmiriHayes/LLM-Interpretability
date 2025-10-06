from transformers import BertTokenizer
import numpy as np
from typing import Tuple

def collaboration_attention_pattern(sentence: str, tokenizer: BertTokenizer) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume sentence splitting respects tokens corresponding approximately
    # To coordination or collaboration context involvement
    keywords = {'and', 'with', 'because', ',', '.', 'to', 'for'}

    # Collect token positions for special tokens
    special_tokens = set(tokenizer.all_special_ids)

    # Iterate over tokens
    for idx, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if token in keywords:
            # Attend to all other tokens within the sentence with a constant weight
            for j in range(1, len_seq - 1):  # Avoid CLS and SEP
                out[idx, j] = 1

    # Normalize the attention scores
    out = out / out.sum(axis=1, keepdims=True)

    # Ensure no NaN values
    out = np.nan_to_num(out)

    return "Coordination and Collaboration between Actions and Subjects", out