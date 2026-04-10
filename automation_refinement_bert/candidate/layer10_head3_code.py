from transformers import PreTrainedTokenizerBase
import numpy as np
import re

# Function to string split keeping coordinates together
# Based on observation of coordinate patterns (e.g., `$ ( 2 , - 6 ) $`) in sentences

def extract_coordinates(sentence):
    # Regex to capture coordinates
    pattern = re.compile(r'\$ \( \(?\-?\d+\) , \(?\-?\d+\) \) \$')
    return re.findall(pattern, sentence)

def coordinate_based_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    coordinates = extract_coordinates(sentence)
    if not coordinates:
        return "No Coordinates Pattern Found", out
    for coord in coordinates:
        # Find the token indexes for the coordinates
        coord_tokens = tokenizer([coord], return_tensors="pt")
        coordinate_indexes = [idx+1 for idx, token_id in enumerate(toks.input_ids[0]) if token_id in coord_tokens.input_ids[0][1:-1]]

        # Apply strong self attention within the coordinates
        for i in coordinate_indexes:
            for j in coordinate_indexes:
                out[i, j] = 1

    out[0, 0] = 1  # For [CLS] token attention
    out[-1, 0] = 1  # For [SEP] token attention

    # Normalize the out matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Coordinate-based Lexical Attention", out