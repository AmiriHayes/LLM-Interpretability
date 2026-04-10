import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

class LexicalAssociationAttention:
    def create_attention_matrix(self, sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
        toks = tokenizer([sentence], return_tensors="pt")
        len_seq = len(toks.input_ids[0])
        out = np.zeros((len_seq, len_seq))

        # Sample hypothetical pattern: identify repetition of keywords (e.g., function names)
        tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
        unique_tokens = set(tokens)

        # Map to track last occurences of unique tokens
        last_occurrence = {}

        # Iterate through tokens to populate attention pattern based on repeated lexical items
        for i, tok in enumerate(tokens):
            if tok in unique_tokens:
                # Mark attention from current token to its last occurrence, if it has one
                if tok in last_occurrence:
                    out[i, last_occurrence[tok]] = 1
                # Update or add current position of token
                last_occurrence[tok] = i

        # CLS and EOS token attention
        out[0, 0] = 1  # CLS to CLS
        out[-1, 0] = 1  # EOS to CLS

        # Normalize each row to sum to 1
        out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1, a_max=None)

        return "Lexical Association Attention", out