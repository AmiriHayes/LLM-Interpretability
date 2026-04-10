from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# This function identifies important or salient words in a sentence based on the given pattern
# from the data, which suggests that certain words, like keywords or focused subject matter
# (e.g., 'needle', 'Lily'), attract attention significantly.
def salient_keywords_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:

    # Tokenizing the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Splitting the sentence and processing for keyword detection
    words = sentence.split()
    # Simple rule to detect potential keywords, focusing on nouns and verbs
    # Noun-like words receive higher attention weights as important elements
    potential_keywords = [word.lower() for word in words if word[0].isupper() or len(word) > 4]

    # Organizing attention in the matrix largely centered around these detected keywords
    for i, tok in enumerate(words):
        if tok.lower() in potential_keywords:
            for j in range(1, len_seq-1):
                out[i+1, j] = 1.0

    # Ensuring the output matrix includes attention compensation
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix, simulating softmax behavior for focus meaning
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows
    return "Keyword and Salient Element Detection", out