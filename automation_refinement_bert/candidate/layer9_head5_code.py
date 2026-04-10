import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")

def coreferential_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(" ".join(words))

    # Create a map from spaCy token to BERT tokens
    id_map = {}
    for tok in toks.input_ids[0]:
        tok_text = tokenizer.decode(tok)
        for word in doc:
            if word.text == tok_text.strip():
                id_map[word.i] = tok.item()
                break

    for coref in doc._.coref_clusters:
        for coref_token in coref.main:
            for ref_token in coref.mentions[1:]:
                coref_start = coref_token.start
                ref_start = ref_token.start
                # Ensure non-empty sequences
                if coref_start in id_map and ref_start in id_map:
                    coref_index = id_map[coref_start]
                    ref_index = id_map[ref_start]
                    out[coref_index, ref_index] = 1
                    out[ref_index, coref_index] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coreferential Attention", out