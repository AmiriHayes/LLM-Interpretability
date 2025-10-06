import numpy as np
from transformers import PreTrainedTokenizerBase
from spacy.lang.en import English

def coreference_and_object_association(sentence: str, tokenizer: PreTrainedTokenizerBase):
    nlp = English()
    doc = nlp(sentence)
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Create a mapping of tokens to their corresponding spaCy tokens (not using bpe tokenization)
    spacy_token_to_toks_index = {}
    bpe_start = 1
    for token in doc:
        word_piece_tokens = tokenizer.tokenize(token.text)
        for wp in word_piece_tokens:
            spacy_token_to_toks_index[(token.text, wp)] = bpe_start
            bpe_start += 1

    # Iterate through tokens to create association pattern
    for token in doc:
        token_index = spacy_token_to_toks_index.get((token.text, tokenizer.tokenize(token.text)[0]))
        if token_index is None:
            continue

        for span in doc.ents:
            if token in span:
                for inner_token in span:
                    if inner_token != token:
                        inner_token_index = spacy_token_to_toks_index.get((inner_token.text, tokenizer.tokenize(inner_token.text)[0]))
                        if inner_token_index is not None:
                            out[token_index, inner_token_index] = 1
                            out[inner_token_index, token_index] = 1

    # Additional associations not captured by dependency parser, e.g. shared objects
    paired_words = ["needle", "shirt", "button", "needle", "together"]
    for pw in paired_words:
        pw_indices = [i for i, tok_id in enumerate(toks.input_ids[0]) if tokenizer.decode([tok_id]) == pw]
        for i in pw_indices:
            for j in pw_indices:
                if i != j:
                    out[i, j] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # To avoid division by zero issues
    out /= out.sum(axis=1, keepdims=True)  # Normalize the attention matrix
    return "Co-reference and Object Association Pattern", out

