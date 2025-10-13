import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import re

def pronoun_focus_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into words
    words = sentence.split()

    # Find the first pronoun or noun phrase focus in the sentence
    focus_word_index = 0
    focus_found = False
    pronoun_regex = re.compile(r'\b(I|he|she|it|they|we|you|me|him|her|us|them)\b', re.IGNORECASE)

    for i, word in enumerate(words):
        if pronoun_regex.match(word):
            focus_word_index = i
            focus_found = True
            break
        if not focus_found and word[0].isupper():  # Assume first capitalized word is a potential noun focus
            focus_word_index = i
            focus_found = True

    # Use tokenizer word_ids to map sentence index to tokens
    word_ids = toks.word_ids(batch_index=0)
    focus_token_index = word_ids.index(focus_word_index)

    # Set attention to focus on the word and its nearby context
    for i in range(len_seq):
        if word_ids[i] is not None:
            out[i, focus_token_index] = 1

    # Ensure CLS, SEP (or equivalent) tokens attend to themselves
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Pronoun Reference and Noun Phrase Focus", out