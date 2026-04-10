import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_boundary_and_action_nouns(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spacy to get tokens and their dependencies
    doc = nlp(sentence)

    # Align SpaCy tokens with tokenizer tokens
    token_mapping = {}
    token_index = 0
    for i, token in enumerate(doc):
        while token_index < len_seq - 2:  # Exclude <s> and </s>
            piece_id = toks.input_ids[0][token_index + 1].item()
            piece_text = tokenizer.decode([piece_id]).strip()
            if piece_text == token.text:
                token_mapping[i] = token_index + 1
                token_index += 1
                break
            else:
                token_index += 1

    # Focus attention on CLS and SEP tokens
    out[0, 0] = 1  # <s> attends to itself strongly
    out[len_seq-1, len_seq-1] = 1  # </s> attends to itself strongly

    # Focus on nouns and action verbs with higher weights
    for token in doc:
        if token.pos_ in {'NOUN', 'VERB'}:
            token_idx = token_mapping.get(token.i)
            if token_idx:
                out[token_idx, 0] = 0.2  # Attention from noun/verb to <s>
                out[token_idx, len_seq-1] = 0.2  # Attention to </s>

    # Normalize attention scores for each token
    for i in range(len_seq):
        if out[i].sum() == 0:
            out[i, -1] = 1
        out[i] /= out[i].sum()  # Normalize row

    return "Focus on Sentence Boundaries and Main Action Nouns", out