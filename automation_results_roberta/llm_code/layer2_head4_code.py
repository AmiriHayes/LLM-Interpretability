from typing import Tuple
from transformers import PreTrainedTokenizerBase
import numpy as np

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention scores to the start token <s> for each word
    for i in range(1, len_seq):  # Skip the first token <s> intentionally as it looks mostly at itself
        out[i, 0] = 1.0
        out[0, 0] = 1.0  # Ensure the start token also looks at itself

    # Add attention to the end token </s>
    out[-1, 0] = 1.0

    # Normalize attention weights for each token to sum to 1
    out = out / out.sum(axis=1, keepdims=True)
    return 'Sentence Beginning Attention Pattern', out