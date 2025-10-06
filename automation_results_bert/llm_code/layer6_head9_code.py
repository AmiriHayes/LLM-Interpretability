import numpy as np
from transformers import PreTrainedTokenizerBase

def complement_structure_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simplified process simulating patterns seen in data
    words = sentence.split()
    complement_triggers = {"with": ["to", "in", "on"], "because": ["was", "had"], "can": ["share", "fix"]}
    sentence_attention_mapping = {w: i + 1 for i, w in enumerate(words)}

    for complement, linked_words in complement_triggers.items():
        if complement in sentence_attention_mapping:
            complement_index = sentence_attention_mapping[complement]
            for linked_word in linked_words:
                if linked_word in sentence_attention_mapping:
                    linked_word_index = sentence_attention_mapping[linked_word]
                    out[complement_index, linked_word_index] = 1
                    out[linked_word_index, complement_index] = 1

    # Assign self-attention to CLS and SEP tokens
    out[0, 0] = 1  # [CLS]
    out[-1, -1] = 1  # [SEP]

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Complement Structure Dependency Pattern", out