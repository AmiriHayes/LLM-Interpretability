from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy's English tokenizer
nlp = spacy.load("en_core_web_sm")

def conjunction_and_coordination_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence using spaCy
    doc = nlp(sentence)

    # Map spaCy token indices to tokenizer indices
    spacy_to_tokenizer = {}
    tok_idx = 1  # The first token, [CLS], is at index 0
    for spacy_token in doc:
        spacy_to_tokenizer[spacy_token.i] = tok_idx
        tok_idx += len(tokenizer.tokenize(spacy_token.text)) or 1

    # Identify conjunction relations and coordinate elements
    for token in doc:
        # Check for conjunctions or coordinating conjunctions
        if token.dep_ == "cc" or token.dep_ == "conj":
            # Attention to the head of the coordination
            head_idx = token.head.i
            if head_idx in spacy_to_tokenizer and token.i in spacy_to_tokenizer:
                head_tok_idx = spacy_to_tokenizer[head_idx]
                tok_idx = spacy_to_tokenizer[token.i]
                out[head_tok_idx, tok_idx] = 1.0
                out[tok_idx, head_tok_idx] = 1.0

    # Ensuring each token attends to itself and others
    for i in range(len_seq):
        out[i, 0] = 1.0  # [CLS] token
        if out[i].sum() == 0:
            out[i, -1] = 1.0  # Attend to [SEP] if no other attention is set

    # Normalize output
    out = out / out.sum(axis=1, keepdims=True)

    return "Conjunction and Coordination Pattern", out