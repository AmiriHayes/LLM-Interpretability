import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    token_to_index = {v: k for k, v in enumerate(toks.word_ids(0))}
    words = sentence.split()

    pronouns = {"he", "she", "it", "they", "we", "I", "you", "her", "him", "us", "them"}

    for index, word in enumerate(words):
        word_lower = word.lower().strip('.,!?"')
        if word_lower in pronouns:
            out[index + 1, index + 1] = 1  # Self-attention for pronouns
            # Account for the salience by making the pronoun have attention with earlier important nouns
            max_attention_length = 5  # Hypothetical constraint for how far we look back
            reverse_index = max(0, index - max_attention_length)
            for j in reversed(range(reverse_index, index)):
                if j+1 in token_to_index and words[j][0].isupper():  # Checking if it is a noun
                    out[index + 1, j + 1] = 1  

    # Ensure no rows are fully zeros (except for special tokens which might get filled later)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun Reference and Salience Pattern", out