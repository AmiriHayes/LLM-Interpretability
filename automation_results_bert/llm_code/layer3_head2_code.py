import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def coordinating_conjunction_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse tokens and their parts of speech
    doc = nlp(sentence)
    word_to_idx = {tok.text: idx + 1 for idx, tok in enumerate(doc)}  # account for [CLS] token

    coordinating_conjunctions = {'and', 'or', 'but', 'for', 'nor', 'so', 'yet'}

    for i, token in enumerate(doc):
        if token.text.lower() in coordinating_conjunctions:
            token_idx = word_to_idx.get(token.text, -1)
            if token_idx != -1:
                # Attend to itself
                out[token_idx, token_idx] = 1.0

                # Find previous and next tokens in the sentence to emphasize coordination
                # Look to the left of the coordinating conjunction
                if token_idx > 1:  # Ensure not to go out of bounds
                    out[token_idx, token_idx - 1] = 0.5

                # Look to the right of the coordinating conjunction
                if token_idx < len_seq - 1:  # Again, ensure we remain within bounds
                    out[token_idx, token_idx + 1] = 0.5

    # Ensure [CLS] and [SEP] have some attention
    out[0, 0] = 1.0  # [CLS]
    out[-1, -1] = 1.0  # [SEP]

    # Ensure that no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Fallback attention to [SEP]

    return 'Coordinating Conjunction Emphasis', out