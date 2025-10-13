import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase
nlp = spacy.load('en_core_web_sm')


def coreference_resolution_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy for tokenization and coreference identification
    doc = nlp(sentence)
    token_posmap = {token.i: tok_idx for tok_idx, token in enumerate(doc)}

    # Process coreference using noun chunks
    for chunk in doc.noun_chunks:
        # Map spaCy indices to token indices
        for token in chunk:
            if token.i in token_posmap:
                token_index = token_posmap[token.i]
                head_index = token_posmap[chunk.root.i]  # Head of the noun chunk
                out[token_index + 1, head_index + 1] = 1

    # Ensure no row is all zeros by defaulting unattached tokens to the last token
    for row in range(1, len_seq - 1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out