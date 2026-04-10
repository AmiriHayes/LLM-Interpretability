import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# Function to determine if a word is primarily vivid/imagery-focused
# This is a simplification for demonstration purposes.
def is_vivid(word):
    vivid_descriptors = {"vibrant", "intricate", "mysteries", "fascinating", "towering", "silent", "sparkling", "marvelous", "unique", "majestic", "countless", "peaceful"}
    return word in vivid_descriptors

def detect_vivid_imagery(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])

    # Initialize matrix
    out = np.zeros((len_seq, len_seq))

    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Align tokens between tokenizer and spaCy
    mapping = {}
    spacy_index = 0
    for i, token in enumerate(toks.input_ids[0]):
        while spacy_index < len(doc):
            if toks.text[0][i] in doc[spacy_index].text:
                mapping[i] = spacy_index
                spacy_index += 1
                break
            spacy_index += 1

    # Assign attention based on vividness of a word and its surrounding context
    for i, token_id in enumerate(toks.input_ids[0]):
        spacy_index = mapping.get(i, None)
        if spacy_index is not None and is_vivid(doc[spacy_index].text.lower()):
            for j in mapping:
                if mapping[j] == spacy_index or (abs(i - j) <= 5):
                    out[i, j] = 1.0

    # Normalize output and return
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Vivid Imagery Detection", out