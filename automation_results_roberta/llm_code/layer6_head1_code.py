import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def semantic_sharing_context(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    def match_tokens(tokenized_sentence, phrase):
        phrase_tokens = tokenizer.tokenize(phrase)
        indices = []
        i = 0
        while i < len(tokenized_sentence):
            if (
                tokenized_sentence[i:i+len(phrase_tokens)]
                == phrase_tokens
            ):
                indices.append(list(range(i, i+len(phrase_tokens))))
                i += len(phrase_tokens)
            else:
                i += 1
        return indices

    sentence_to_token_index = {i: i for i in range(len_seq)} # Simple mapping

    relevant_phrases = ["sharing", "together", "share", "with", "fix"]
    for phrase in relevant_phrases:
        indices_group = match_tokens(toks.tokens(), phrase)
        for indices in indices_group:
            for ind in indices:
                for j in indices:  # Attention within phrase
                    out[ind, j] = 1
                for ind2 in indices_group:
                    if ind2 != indices:
                        for j in ind2:  # Cross-cluster attention
                            out[ind, j] = 0.5

    # Ensure attention to CLS and SEP as defaults
    for row in range(1, len_seq-1):
        if out[row].sum() == 0:
            out[row, 0] = 0.1  # slight attention to <s>
            out[row, -1] = 0.9  # higher attention to </s>

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to sum to 1

    return "Semantic Role: Contextual Sharing Cluster", out