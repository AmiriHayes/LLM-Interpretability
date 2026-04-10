import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load English tokenizer and tagger from spaCy
nlp = spacy.load('en_core_web_sm')

def comma_coordination_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)

    # Create a mapping from token indices of tokenizer to spaCy
    word_to_token = {}
    tokenizer_index = 1  # Start after [CLS]
    for i, token in enumerate(doc):
        token_text = token.text
        # some elements of the tokenizer will split into sub-words starting with '##'
        sub_tokens_count = sum(t.startswith('##') for t in tokenizer.tokenize(token_text))
        for _ in range(sub_tokens_count + 1):  # Add at least for the token itself
            word_to_token[i] = tokenizer_index
            tokenizer_index += 1

    # Check token associations with commas
    for token in doc:
        if token.pos_ == 'PUNCT' and token.text == ',':
            # Usually, the commas coordinate separate element lists
            comma_idx = word_to_token.get(token.i, None)

            if comma_idx is not None:
                # Find connected elements before and after the comma
                left_idx = None
                right_idx = None

                # Seek leftwards for a non-punct token
                for i in range(token.i - 1, -1, -1):
                    if doc[i].pos_ != 'PUNCT':
                        left_idx = i
                        break

                # Seek rightwards for a non-punct token
                for i in range(token.i + 1, len(doc)):
                    if doc[i].pos_ != 'PUNCT':
                        right_idx = i
                        break

                if left_idx is not None and right_idx is not None:
                    left_token_idx = word_to_token[left_idx]
                    right_token_idx = word_to_token[right_idx]
                    out[comma_idx, left_token_idx] = 1
                    out[comma_idx, right_token_idx] = 1

    out[0, 0] = 1  # CLS token
    out[-1, 0] = 1  # SEP token
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Comma Coordination Linkage", out