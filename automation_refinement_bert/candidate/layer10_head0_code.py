import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
from typing import Tuple

# Load spaCy English model for linguistic features
nlp = spacy.load('en_core_web_sm')

def adjective_noun_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spaCy
    doc = nlp(sentence)
    token_indices = {token.i: idx for idx, token in enumerate(doc)}

    # Find Adjective-Contextual Adjective relationships
    for token in doc:
        if token.pos_ in ['ADJ', 'NOUN']:
            attention_indices = [
                token_indices[adj.i] + 1
                for adj in token.lefts
                if adj.pos_ == 'ADJ'
            ]
            # Add attention
            if attention_indices:
                base_index = token_indices[token.i] + 1
                for idx in attention_indices:
                    out[base_index, idx] = 1.0
                    out[idx, base_index] = 1.0

    # Ensure self-attention
    for i in range(len_seq):
        out[i, i] = 1.0

    # Normalize the attention matrix
    out = (out + 1e-4) / out.sum(axis=1, keepdims=True)

    return "Adjective-Noun Grouping", out