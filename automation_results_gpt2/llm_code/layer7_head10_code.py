from transformers import PreTrainedTokenizerBase
import numpy as np


def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy and make sure tokens match
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Create a mapping from spaCy tokens to BERT tokens
    tok_alignment = {i: toks.input_ids[0][i].item() for i in range(len_seq - 1)}

    # Given example data where coref is likely between subjects and their associated verbs/nouns
    coref_map = {}
    for token in doc:
        if token.pos_ in {"PRON", "PROPN", "NOUN"}:
            coref_map[token.i] = token

    # Assigning attentions among references found
    for i in coref_map:
        ref_id = tok_alignment.get(i + 1, None)
        if ref_id:
            for j in range(len_seq):
                out[i + 1, j] = 1.0  # the +1 accounts for the [CLS] token

    # Normalize the output matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensuring no row is without attention focus
        out += 1e-4
        out /= out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Coreference Resolution Pattern", out