import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to capture Segment Boundary Attention Pattern
def segment_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # It seems the model attends primarily to sentence boundaries
    # Set strong attention to the first and last token of the sentence
    out[0, 0] = 1  # CLS token
    out[-1, -1] = 1  # SEP or end token (.)

    # Assuming strong self-attention and secondary sentence parts
    for i in range(1, len_seq - 1):
        # Self attention
        out[i, i] = 0.5

        # Strong attention to both ends as they act like boundaries
        out[i, 0] = 0.25  # CLS
        out[i, -1] = 0.25  # SEP/end token

    # Normalize each row to simulate attention distribution in transformer
    out += 1e-4  # Add small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return 'Segment Boundary Attention', out