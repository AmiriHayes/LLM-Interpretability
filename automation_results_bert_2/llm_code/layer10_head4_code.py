import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Map spaCy tokens to tokenizer tokens
    word_mapping = dict()
    token_start_index = 0
    spacy_words = sentence.split()  # crude split, should be refined with proper NLP library
    for idx, word in enumerate(spacy_words):
        toks_for_word = tokenizer([word], return_tensors="pt").input_ids[0][1:-1]
        tok_range = range(token_start_index + 1, token_start_index + 1 + len(toks_for_word))
        word_mapping.update({tok_idx: idx for tok_idx in tok_range})
        token_start_index += len(toks_for_word)

    # List of common punctuation characters
    punctuation_tokens = [",", ".", "!", "?", "'", "\"", "(", ")"]

    # Providing higher attention to punctuation tokens
    for i in range(1, len_seq - 1):
        tok = toks.input_ids[0][i].item()
        tok_str = tokenizer.convert_ids_to_tokens(tok)
        if tok_str in punctuation_tokens:
            # Attention pattern favoring specific punctuation
            out[i] = 1 / (len_seq - 2)  # Distributes even attention to all tokens
            out[i][i] = 0  # Self-attention is weak
            out[i][0] = out[i][-1] = 0  # Avoid focusing on CLS and SEP

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Punctuation Focus", out