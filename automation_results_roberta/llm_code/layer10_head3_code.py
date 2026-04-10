import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def end_of_sentence_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to start-of-sentence and end-of-sentence tokens
    out[0, :] = 1.0 # Self attend to the start token <s>
    out[-1, :] = 1.0 # Self attend to the end token </s>

    for i in range(1, len_seq-1):
        # Assign partial attention to the second-to-last token typically being the punctuation
        out[i, -1] = 0.5
        out[i, i] = 0.5 # Self attention among sentence tokens

    return 'End-of-Sentence and Spann Attentiveness', out