import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')


def subject_focus_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spacy for linguistic analysis
    doc = nlp(sentence)
    spacy_tokens = [token.text for token in doc]
    tok_to_spacy_index = {i: token.i for i, token in enumerate(doc)}

    # Initially focus on the pronouns (assuming they are subjects)
    for token in doc:
        if token.pos_ == 'PRON' and (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass'):
            spacy_index = token.i
            for tok_idx in range(len_seq):
                if tok_idx in tok_to_spacy_index and tok_to_spacy_index[tok_idx] == spacy_index:
                    out[tok_idx, tok_idx] = 1  # Assign high attention to subject pronoun

                    # Give relatively high attention to first few tokens (beginning of sentence)
                    for j in range(min(5, len_seq)):
                        out[tok_idx, j] = 0.5  # A weaker relative attention for initial tokens

    # Ensure there's at least some attention to the final token (often a punctuation in GPT-style tokenization)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Subject Pronoun/Part-of-Sentence Focus", out