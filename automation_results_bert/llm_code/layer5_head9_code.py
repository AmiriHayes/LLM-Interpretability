import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# Entity Extraction and Relationship Mapping
# Key pattern observed: The attention focuses on possessive relations or assignments e.g., 'her mom', 'their needle',
# capturing a relationship of ownership or intimacy between subjects or objects.
def ownership_relation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)

    # Extract POS tags and dependency parsing (optional) from spaCy
    poss_relationships = []
    for token in doc:
        if token.dep_ == "poss":  # 'poss' denotes possessive
            possessor_idx = token.i  # Spacy index
            possession_idx = token.head.i
            poss_relationships.append((possessor_idx, possession_idx))

    # Map Spacy indices to Tokenizer output
    token_index_map = []
    for i, tok in enumerate(doc):
        start = tok.idx  # Starting character idx of token
        token_index_map.append((i, start, start + len(tok.text)))

    # Tokenizer token alignment mapping
    spacy_to_tokenizer_map = {}
    for i_spacy, ichar_start, ichar_end in token_index_map:
        for token_idx, input_id in enumerate(toks["input_ids"][0][1:-1]):  # avoid CLS and SEP
            text = tokenizer.decode([input_id])
            if (ichar_start <= token_idx < ichar_end) and text.strip():
                spacy_to_tokenizer_map[i_spacy] = token_idx + 1  # Account for CLS offset

    # Fill attention based on possession
    for possessor_idx, possession_idx in poss_relationships:
        if possessor_idx in spacy_to_tokenizer_map and possession_idx in spacy_to_tokenizer_map:
            attn_possessor = spacy_to_tokenizer_map[possessor_idx]
            attn_possession = spacy_to_tokenizer_map[possession_idx]
            out[attn_possessor, attn_possession] = 1
            out[attn_possession, attn_possessor] = 1

    # Normalize attention matrix
    for row in range(len_seq):
        row_sum = out[row].sum()
        if row_sum == 0:
            out[row, -1] = 1.0  # assign to [SEP] if no attention
        else:
            out[row] /= row_sum

    return "Possessive or Ownership Relationship", out