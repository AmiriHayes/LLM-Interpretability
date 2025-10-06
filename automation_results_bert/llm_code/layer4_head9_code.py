import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def conjunction_clause_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Perform NLP analysis using spaCy to identify conjunctions and coordinated clauses
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    sentence_text = " ".join(words)
    doc = nlp(sentence_text)

    # Identify conjunctions and coordinated clauses
    conjunctions = []
    for idx, token in enumerate(doc):
        if token.dep_ in ['cc', 'conj']:
            conjunctions.append(idx + 1)  # +1 to adjust the indices for special tokens

    # Create attention patterns based on identified conjunctions
    for conjunction_idx in conjunctions:
        for token_idx in range(len_seq):
            if token_idx != conjunction_idx:
                out[conjunction_idx, token_idx] = 1
                out[token_idx, conjunction_idx] = 1

    # Ensure out has no row of zeros (attention to SEP token)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to [SEP]

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Conjunction and Coordinated Clause Attention Pattern", out
