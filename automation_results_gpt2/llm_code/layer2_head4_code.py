import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase
nlp = spacy.load('en_core_web_sm')

def subject_pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy
    doc = nlp(sentence)
    # Create a dictionary to align spacy tokens to the bpe tokens
    bpe_to_word_map = {}
    for token in doc:
        for i in range(len(toks.input_ids[0])):
            if token.text.strip() == toks.tokens(0)[i]:
                bpe_to_word_map[i] = token.i
                break

    # Find the subject pronoun
    for i, token in enumerate(doc):
        if token.dep_ in {'nsubj', 'nsubjpass'} and token.pos_ in {'PRON'}:
            subj_index = i
            # Ensure other tokens receive minimal attention from subject pronoun
            for j in range(len_seq):
                out[j, subj_index] = 1 if j == subj_index else 0

    return "Subject Pronoun Resolution", out