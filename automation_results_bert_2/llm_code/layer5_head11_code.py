import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")

# Function to predict attention pattern based on coordinated conjunctions

def coordinated_conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spaCy to align with BERT tokens
    # Capture the offset of actual words to BERT tokenization
    word_ids = toks.word_ids(batch_index=0)
    words = [toks[i] for i in range(len_seq)]
    doc = nlp(" ".join(words))

    # Identify coordinated conjunctions
    for token in doc:
        if token.pos_ == "CCONJ":
            idx = word_ids.index(token.i) if token.i in word_ids else -1
            for child in token.children:
                child_idx = word_ids.index(child.i) if child.i in word_ids else -1
                if child.dep_ in ("conj", "cc"):  # Check for conjunction dependencies
                    out[idx, child_idx] = 1
                    out[child_idx, idx] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to [SEP] token as default

    return "Coordinated Conjunction Attention Pattern", out

