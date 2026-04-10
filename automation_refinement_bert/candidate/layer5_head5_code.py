import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to calculate Paired Anchor Attention
# Tokens are particularly attracted to paired anchors such as ',' and 'and', or similar patterns
# observed often in the attention data.
def paired_anchor_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence on spaces to get the words
    words = sentence.split()

    # Tokens that might serve as anchors in pairs
    anchor_tokens = {',': 'and', ';': 'and', ':': 'and'}

    # Iterate through sentence to find anchor-like pairs
    for i, word in enumerate(words):
        if word in anchor_tokens:
            # Find the position of the paired anchor if it exists
            sibling_word = anchor_tokens[word]
            if sibling_word in words[i + 1:]:
                sibling_index = words.index(sibling_word, i + 1)
                # Mark in the attention matrix
                out[i + 1, sibling_index + 1] = 0.5  # Slight broader attention from the anchor itself
                out[sibling_index + 1, i + 1] = 0.5
                for j in range(i + 2, sibling_index):
                    out[i + 1, j + 1] = 0.3
                    out[j + 1, sibling_index + 1] = 0.3

    # Ensure no row has all zeros (focused attention to [SEP], often the last token)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0  # Normally the [SEP] token

    return 'Paired Anchor Attention', out