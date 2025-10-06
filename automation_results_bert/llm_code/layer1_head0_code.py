from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def verb_centric_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Pre-process the sentence with spaCy to extract verbs
    doc = nlp(sentence)
    verbs_index = []
    # Use dictionary to align spaCy tokens with tokenizer tokens
    alignment = {}
    token_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)

    # Create a mapping from spaCy tokens to tokenizer tokens
    token_map = {}
    current_spacy_index = 0
    for i, wi in enumerate(word_ids):
        if wi is not None:
            alignment[i] = current_spacy_index
            current_spacy_index += 1

    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            verbs_index.append(token.i)

    # Assign higher attention weights to verbs and their related indices in tokenized format
    for i in range(1, len_seq-1):
        if i in alignment:
            spacy_index = alignment[i]
            if spacy_index in verbs_index:
                out[i, :] = 1
                out[:, i] = 1

    # Ensure no row is all zeros by assigning some attention to the [CLS] that has index 0
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    return "Verb-Centric Attention Pattern", out