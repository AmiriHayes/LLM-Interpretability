import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Load the spaCy English model to process the sentence
nlp = spacy.load('en_core_web_sm')


def connector_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence using spaCy to obtain token information
    doc = nlp(sentence)

    # Map token positions from spaCy to the tokenizer output
    tok_map = {}
    spacy_idx = 0
    for tok_idx, word_id in enumerate(toks.word_ids(batch_index=0)):
        if word_id is not None:
            tok_map[word_id] = tok_idx

    # Analyze attention patterns
    connectors = {"and", "but", "or", "because", "so", "although", ","}

    for token in doc:
        if token.text.lower() in connectors:
            token_idx = token.i
            if token_idx in tok_map:
                attn_idx = tok_map[token_idx]
                # Highlighting conjunctions and their immediate linguistic partners
                for child in token.children:
                    if child.i in tok_map:
                        child_idx = tok_map[child.i]
                        out[attn_idx, child_idx] = 1.0
                        out[child_idx, attn_idx] = 1.0
                # Also link to the previous main verb or noun (default behavior)
                head_idx = tok_map.get(token.head.i, None)
                if head_idx is not None:
                    out[attn_idx, head_idx] = 1.0
                    out[head_idx, attn_idx] = 1.0

    # Ensure every token at least attends to itself
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    # Normalize to keep attention probabilistic
    out = out / out.sum(axis=1, keepdims=True)
    return "Conjunction or Connector Pattern", out