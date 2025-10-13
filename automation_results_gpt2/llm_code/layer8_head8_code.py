import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def subject_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into words
    words = sentence.split()

    # Define a heuristic for identifying the subject phrase
    # Assumption: The subject is likely at the start of the sentence and is a noun phrase
    subject_indices = set()
    current_index = 1  # Start after [CLS]

    noun_tag = ["NN", "NNS", "NNP", "NNPS"][0] # replace with proper POS

    # Simulating a simple subject-phrase identification using manual rules
    # This is a simplistic check assuming subject is one of first few tokens
    # Extract 2 tokens from the start or till an explicit punctuation
    for i, word in enumerate(words):
        if i >= 2: # simple rule to capture 'subject' like phrases in first two words
            break
        if i == 0 or word.isalnum():
            subject_indices.add(current_index)
        current_index += 1

    # Fill the attention matrix for predicted subject-phrase attention
    for subject_index in subject_indices:
        out[subject_index, :] = 1  # The subject attends to every word
        out[:, subject_index] = 1  # Every word attends to the subject

    # Ensure no row is zero by attending to [SEP] (or the last token)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Subject-Phrase Attention Pattern", out