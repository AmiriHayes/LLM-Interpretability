import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function based on the hypothesis

# Collocation refers to the tendency for certain words, particularly verb-noun pairs as noted here, to frequently appear together.
# This function identifies collocation patterns in verb-noun/base combinations, likely highlighting verbs and their direct objects
# or subjects that typically co-occur. 

def collocation_verb_noun(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    # Identifying simple verb-noun collocations
    for i in range(1, len(words) - 1):
        if 'ed' in words[i] or 'ing' in words[i] or words[i] in ['was', 'were', 'is', 'are', 'be', 'been', 'am']:
            if i > 0 and words[i - 1] not in ['the', 'a', 'an'] and words[i - 1].islower():
                # Relationships to noun before verb (e.g., subject-verb or preparatory position)
                out[i, i - 1] = 1
            if i < len(words) - 1 and words[i + 1].islower():
                # Relationships to noun after verb (e.g., verb-object)
                out[i, i + 1] = 1

    for row in range(len_seq):
        # Ensure that no row is all zeros by attending to SEP token
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Collocation and Verb-Noun Pairing", out