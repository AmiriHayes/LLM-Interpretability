from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')

def coordination_structure(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    doc = nlp(sentence)
    # Extract tokens with their indices
    word_to_token_map = dict(enumerate(words))

    # For looking at coordination structure, we assume a focus on conjunctions
    conjunctions = {"and", "or", "but", "nor", "for", "yet", "so"}
    coord_indices = [i for i, token in enumerate(doc) if token.text in conjunctions]

    for i in coord_indices:
        for j in range(i+1, len_seq):
            if words[j] == '[SEP]':
                break
            out[i, j] = 1

    for i in coord_indices:
        out[i, i] = 1 # self-reference in case of no coordinates

    # Normalize output (add epsilon for numerical stability)
    out += 1e-5
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination Structure Focus", out

