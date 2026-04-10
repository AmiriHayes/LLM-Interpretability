from transformers import PreTrainedTokenizerBase
import numpy as np


def named_entities_and_concepts(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simulating recognition of named entities and key concepts
    words = sentence.split()

    # Mock detection of entities/concepts based on word position
    highlighted_indices = set()

    # Scan the sentence for some key concepts (mock)
    for i, word in enumerate(words):
        if word.lower() in {"lily", "mom", "needle", "shirt"}:
            highlighted_indices.add(i + 1)  # account for <s> token in offset

    # Fill in attention pattern based on detected entities/concepts
    for i in highlighted_indices:
        for j in highlighted_indices:
            out[i, j] = 1  # Self-attention between key concepts

    # Ensure matrix rows have at least one non-zero entry
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Add attention to the end of sentence token

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Focus on Named Entities and Key Concepts", out