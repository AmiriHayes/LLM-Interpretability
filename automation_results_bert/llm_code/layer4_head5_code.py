import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def connect_modality_and_adverbials(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis: Head focuses between modals such as 'could', 'would', 'can', etc., and adverbial clauses
    modal_tokens = {'can', 'could', 'would', 'might', 'must', 'should'}
    adverbial_triggers = {'because', 'so', 'and', 'but'}

    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    adverbial_indices = [i for i, token in enumerate(decoded_tokens) if token in adverbial_triggers]
    modal_indices = [i for i, token in enumerate(decoded_tokens) if token in modal_tokens]

    for modal_idx in modal_indices:
        # Connect modal verbs to logical adverbs in adverbial triggers
        for adv_idx in adverbial_indices:
            out[modal_idx, adv_idx] = 1
            out[adv_idx, modal_idx] = 1

    # Ensure each token has an attention on either itself or SEP token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Modality and Adverbial Clausal Connection', out