import numpy as np
from transformers import PreTrainedTokenizerBase

# Define a list of pronouns
pronouns = {'I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves'}

# Function for cohesion pattern

def pronoun_cohesion(sentence: str, tokenizer: PreTrainedTokenizerBase) -> "Tuple[str, np.ndarray]":
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get tokens from the sentence
    word_offsets = [(tokenizer.convert_ids_to_tokens(idx.item()), idx.item()) for idx in toks.input_ids[0]]

    # Loop through the sentence to find pronouns
    for idx, (token, _id) in enumerate(word_offsets):
        if token in pronouns:
            for jdx, (comp_token, comp_id) in enumerate(word_offsets):
                if comp_token == token or comp_token in pronouns:
                    out[idx, jdx] = 1

    # Make sure every token focuses on at least one other token.
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Return pattern name and the computed matrix
    return "Self-Referential and Pronoun Cohesion", out