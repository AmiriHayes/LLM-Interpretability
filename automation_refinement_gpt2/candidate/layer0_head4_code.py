import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

def describe_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    nlp = spacy.load('en_core_web_sm')
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token map to account for any tokenization differences between spaCy and the transformer tokenizer
    tok_idx_map = {}
    spacy_tok_idx = 0
    for token in nlp(sentence):
        spacy_text = token.text
        while tok_idx_map.get(spacy_tok_idx) is None:
            transformer_tok_ids = toks.input_ids[0][spacy_tok_idx:spacy_tok_idx + len(spacy_text)]
            toks_in_str = tokenizer.decode(transformer_tok_ids).strip()
            if toks_in_str == spacy_text or spacy_text in toks_in_str:
                tok_idx_map[spacy_tok_idx] = token.idx
            spacy_tok_idx += 1

    # Adjectives and nouns extraction from spacy and alignment
    doc = nlp(sentence)
    adj_idx = [token.i for token in doc if token.pos_ == 'ADJ']
    noun_idx = [token.i for token in doc if token.pos_ == 'NOUN']

    # Build the attention pattern
    for i in range(len_seq):
        for adj in adj_idx:
            for noun in noun_idx:
                if abs(adj - noun) < 5:  # Only focus attention on closely related words (contextual proximity)
                    out[tok_idx_map.get(adj, adj), tok_idx_map.get(noun, noun)] = 1
                    out[tok_idx_map.get(noun, noun), tok_idx_map.get(adj, adj)] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Focus on Descriptive Elements", out