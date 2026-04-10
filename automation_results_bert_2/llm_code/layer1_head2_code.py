from transformers import PreTrainedTokenizerBase
import numpy as np

def pronoun_referent_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Predict attention pattern based on pronoun referents."""
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Dictionary mapping tokens to positions
    token_mapping = {idx: toks.input_ids[0][idx].item() for idx in range(len_seq)}

    # Find pronouns
    for idx, token_id in token_mapping.items():
        token = tokenizer.decode(token_id)
        if token.lower() in {"he", "she", "it", "they", "her", "him", "them"}:
            # Find referent - assume the referent is the nearest noun before the pronoun
            for j in reversed(range(1, idx)):
                prev_token = tokenizer.decode(token_mapping[j])
                # Consider nouns (simplified as capitalized words in this simple logic)
                if prev_token.istitle():
                    # Sample attention to referent
                    out[idx, j] = 1.0
                    break

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Pronoun Referent Pattern", out