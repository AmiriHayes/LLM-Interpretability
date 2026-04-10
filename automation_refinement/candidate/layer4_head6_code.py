import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Loading the spaCy model for linguistic processing
nlp = spacy.load('en_core_web_sm')


def descriptive_phrase_coupling(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    # Using SpaCy to get dependency parses
    doc = nlp(" ".join(words))

    # Create a dictionary mapping token indices between tokenizer and spacy for alignment
    word_to_token = {}
    spacy_index = 0
    for tok in toks["input_ids"][0]:
        current_token = tokenizer.decode([tok.item()]).strip()
        if current_token and current_token in doc[spacy_index].text:
            word_to_token[spacy_index] = tok.item()
            spacy_index += 1
            if spacy_index >= len(doc):
                break

    # Identify descriptive phrases and connect their main word
    for chunk in doc.noun_chunks:
        main_word_index = chunk.root.i
        # Connect descriptive phrases within their span in the matrix
        for i in range(chunk.start, chunk.end):
            if i in word_to_token:
                out[word_to_token[i] - 1, word_to_token[main_word_index] - 1] = 1
                out[word_to_token[main_word_index] - 1, word_to_token[i] - 1] = 1

    # CLS and SEP connections
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return "Descriptive Phrase Coupling", out