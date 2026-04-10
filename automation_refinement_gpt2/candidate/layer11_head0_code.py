import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def noun_dominance_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str, np.ndarray:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and parse sentence with spaCy
    doc = nlp(sentence)

    # Find indices of nouns in spaCy and their corresponding GPT-2 token indices
    tok_map = {}
    spacy_index = 0
    for idx, token_id in enumerate(toks.input_ids[0]):
        tok = tokenizer.decode([token_id.item()]).strip()
        while spacy_index < len(doc) and doc[spacy_index].text != tok:
            spacy_index += 1
        tok_map[spacy_index] = idx

    # Add attention for nouns to their surrounding context
    for token in doc:
        if token.pos_ == 'NOUN':
            noun_idx = tok_map[token.i]
            for adj in token.children:
                if adj.dep_ in ('amod', 'compound'):  # Adjective modifiers or noun compound structure
                    adj_idx = tok_map[adj.i]
                    out[noun_idx, adj_idx] = 1
                    out[adj_idx, noun_idx] = 1
            parent = token.head
            if parent.pos_ in ('NOUN', 'PROPN'):  # Parent noun or proper noun
                parent_idx = tok_map[parent.i]
                out[parent_idx, noun_idx] = 1
                out[noun_idx, parent_idx] = 1

    # Ensure no row is all zeros by adding attention to CLS token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0  # CLS token in GPT-2 is usually first

    return 'Noun Dominance Attention', out