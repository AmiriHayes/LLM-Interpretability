import numpy as np
from transformers import PreTrainedTokenizerBase
import re

def complex_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    # A simple regex pattern to capture compound words (e.g., hyphenated or concatenated forms)
    complex_word_pattern = re.compile(r'(?<!\w)(\w{3,}(?:\#\w+)+)(?!\w)')
    # Find all complex words
    complex_words = [(m.group(), m.start()) for m in complex_word_pattern.finditer(sentence)]

    for word, start_idx in complex_words:
        # Finding the start token index and end token index for each complex word in the sentence
        token_start_index = sum(len(w) + 1 for w in words if sentence.find(w) < start_idx) + 1
        token_end_index = token_start_index + len(tokenizer.tokenize(word))

        # Assign attention pattern
        for i in range(token_start_index, token_end_index):
            out[i, token_start_index:token_end_index] = 1

    # Add attention for special tokens [CLS] and [SEP]
    out[0, 0] = 1  # [CLS]
    out[-1, 0] = 1  # [SEP]

    # Normalize out matrix by row to emulate attention softmax behavior
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / np.maximum(row_sums, 1.0)

    return "Complex Word Attention", out