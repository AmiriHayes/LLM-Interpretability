import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def quote_speech_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    sentence_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    doc = nlp(sentence)

    # Prepare mapping from token positions to words
    word_to_token_idx = {}
    token_to_word_idx = {}
    word_idx = 0
    token_idx = 0
    for word in doc:
        while token_idx < len(sentence_tokens):
            token = sentence_tokens[token_idx]
            if word.text.startswith(token.replace('##', '')):
                word_to_token_idx[word_idx] = token_idx
                token_to_word_idx[token_idx] = word_idx
                token_idx += 1
                break
            token_idx += 1
        word_idx += 1

    # Find indices of quotes and related speech tags
    quote_indices = []
    speech_indices = []
    for tok in doc:
        if tok.text in {'"', '\'', '``', "''", '`', '”', '“', '‘', '’'}:
            quote_indices.append(word_to_token_idx.get(tok.i))
        if 'VB' in tok.tag_ and tok.dep_ == 'ROOT':  # Looking for speech-like verbs
            for child in tok.children:
                if child.dep_ in {'nsubj', 'nsubjpass'}:
                    speech_indices.append(word_to_token_idx.get(child.i))

    # Set attention for quote and speech tags
    for i in quote_indices:
        if i is not None:
            out[i, i] = 1
    for i in speech_indices:
        if i is not None:
            out[i, i] = 1

    # Normalize attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Quote and Speech Tag Attention", out
