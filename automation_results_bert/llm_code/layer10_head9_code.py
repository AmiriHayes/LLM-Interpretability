import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

def sharing_and_action_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy to align with BERT tokenizer
    doc = nlp(sentence)
    word_indices = {token.idx: i for i, token in enumerate(doc)}
    tok2spacy = {i: word_indices[doc[i].idx] for i in range(len(doc))}

    for i, token in enumerate(doc):
        if "share" in token.lemma_ or "sew" in token.lemma_ or "fix" in token.lemma_: 
            if i+1 < len(tokenizer.tokenize(sentence)):
                out[tok2spacy[i], tok2spacy[i+1]] = 1
            out[tok2spacy[i], tok2spacy[i]] = 1
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sharing and Action Related Focus", out