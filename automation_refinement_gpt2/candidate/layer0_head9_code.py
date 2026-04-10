from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function definition

def subject_centered_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Assuming the first non-function word token often acts as a subject
    potential_subjects = ['NOUN', 'PROPN', 'PRON']
    subject_index = 0
    for i, token in enumerate(decoded_tokens):
        if any(word in token for word in ["I", "you", "he", "she", "it", "we", "they"]):  # Common pronouns as subjects
            subject_index = i
            break
        if token.istitle() and token.isalpha():
            subject_index = i
            break

    # Set full attention from subject-token to others, and others back to the subject
    for i in range(len_seq):
        if i != subject_index:
            out[subject_index, i] = 1.0
            out[i, subject_index] = 1.0

    # Normalize the outputs
    out += 1e-4  # Avoid zero divisions
    out = out / np.sum(out, axis=1, keepdims=True)

    return "Sentence Subject-centered Attention", out