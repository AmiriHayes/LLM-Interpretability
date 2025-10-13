from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def pronoun_noun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Generate alignment of tokens between tokenizer and spaCy
    word_ids = toks.word_ids(batch_index=0)
    spacy_tokens = nlp(sentence)

    # Creating a map for token alignment
    spacy_to_tokenizer = {i: idx for idx, i in enumerate(word_ids) if i is not None}

    # Identify pronouns and noun tokens
    pronoun_indices = {i for i, token in enumerate(spacy_tokens) if token.pos_ == 'PRON'}
    noun_indices = {i for i, token in enumerate(spacy_tokens) if token.pos_ == 'NOUN'}

    # Reflect attention on pronouns and nearby nouns
    for idx in range(1, len_seq - 1):
        if idx in spacy_to_tokenizer:
            spacy_idx = spacy_to_tokenizer[idx]
            if spacy_idx in pronoun_indices:
                for noun_idx in noun_indices:
                    if noun_idx in spacy_to_tokenizer:
                        aligned_idx = spacy_to_tokenizer[noun_idx]
                        out[idx, aligned_idx] = 1.0

    # Ensure CLS and EOS tokens remain self-attentive
    out[0, 0] = 1.0  # CLS
    out[-1, -1] = 1.0  # EOS

    # Normalize each row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()  # Normalize

    return "Pronoun and Noun Cluster Attention", out