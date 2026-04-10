import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy language model
en_nlp = spacy.load('en_core_web_sm')

# Function to capture pronoun resolution pattern for Layer 4, Head 3
def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = en_nlp(sentence)
    token_map = {}

    # Map spacy tokens to BERT tokens
    for token in doc:
        subword_tokens = tokenizer.tokenize(token.text)
        bert_start_idx = len(token_map) + 1  # CLS token at index 0
        for subtoken in subword_tokens:
            token_map[bert_start_idx] = token
            bert_start_idx += 1

    # pattern focuses on pronouns resolving to potential antecedents
    for tok_idx, token in token_map.items():
        if token.pos_ == 'PRON':
            for child in token.head.children:
                if child.dep_ in {'nsubj', 'dobj', 'pobj'}:
                    # Find BERT token index for antecedent token
                    ant_idx = next((k for k, v in token_map.items() if v == child), None)
                    if ant_idx:
                        out[tok_idx, ant_idx] = 1
                        out[ant_idx, tok_idx] = 1

    # Ensure no row is a row of zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Pronoun Resolution Pattern", out