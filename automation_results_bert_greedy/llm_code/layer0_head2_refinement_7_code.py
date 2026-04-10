from transformers import PreTrainedTokenizerBase
import numpy as np

def compound_word_attachment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # For each word pair in the sentence, identify compound words indicated in the data
    # Connect compound parts, e.g., needle+in, ##p+bee, with strong attention
    pairings = [('bee', '##p'), ('se', '##w')]

    # Generate attention based on compound word patterns
    for i in range(1, len_seq-1):
        token_i = tokenizer.convert_ids_to_tokens(toks.input_ids[0][i].item())
        # For each word pair, reinforce the relation
        for pair in pairings:
            if pair[0] in token_i:
                for j in range(i+1, len_seq-1):
                    token_j = tokenizer.convert_ids_to_tokens(toks.input_ids[0][j].item())
                    if pair[1] in token_j:
                        out[i, j] = 1
                        out[j, i] = 1
                        break

    # Default attention for CLS and SEP tokens
    out[0, 0] = 1  # CLS self-attention
    out[-1, -1] = 1  # SEP self-attention

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return 'Compound Word Attachment', out