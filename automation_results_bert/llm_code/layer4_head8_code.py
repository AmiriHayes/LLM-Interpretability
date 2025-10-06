import numpy as np
from transformers import PreTrainedTokenizerBase, AutoTokenizer

# Define the function

def conjunction_relation(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set the attention for conjunctions
    conjunctions = {"and", "because", "so", ",", "but"}

    # Decode tokens to determine conjunction positions
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Iterate over the tokens
    for i, token in enumerate(decoded_tokens):
        current_word = token.lower()

        # Check if the token is a conjunction
        if current_word in conjunctions:
            # Assign attention to the token to the nearest content words
            if i + 1 < len_seq:
                out[i, i+1] = 1  # Attend to next token
            if i - 1 >= 0:
                out[i, i-1] = 1  # Attend to previous token

    # Ensure CLS and SEP have attention to ensure non-zero attention rows
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the output matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize for attention

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction and Relation Pattern", out

# Example usage with the input sentence and a loaded tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# pattern_name, attention_matrix = conjunction_relation("your input sentence here", tokenizer)