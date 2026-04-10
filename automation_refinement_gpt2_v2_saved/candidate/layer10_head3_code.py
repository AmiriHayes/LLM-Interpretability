from transformers import PreTrainedTokenizerBase
import numpy as np
def duplicate_token_suppression(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    seen = set()
    token_id_to_position = {}
    for idx, token_id in enumerate(toks.input_ids[0]):
        if token_id.item() in seen:
            for seen_idx in token_id_to_position[token_id.item()]:
                out[idx, seen_idx] = 1
        else:
            seen.add(token_id.item())
            token_id_to_position[token_id.item()] = []
        token_id_to_position[token_id.item()].append(idx)
    out[0, 0] = 1  # self attention for CLS token
    out[-1, 0] = 1  # attention to EOS token
    out += 1e-4  # small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # row normalization
    return "Duplicate Token Suppression", out