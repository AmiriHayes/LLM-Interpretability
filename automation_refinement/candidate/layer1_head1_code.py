import numpy as np
from transformers import PreTrainedTokenizerBase
import re

# Function to predict pattern based on hypothesized pattern
# Compound-Compound Attention Linking

def compound_compound_attention_linking(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Example pattern: 'compound|compound' or 'compound|##compound' 
    # Create regex to detect compounds connected by '|'
    pattern = re.compile(r'(\w+\|##?\w+)')

    # Tokenize sentence using space split
    # These are human-readable tokens and may not match model tokenizers
    words = sentence.split()

    # Use spacy or any regex rule for formal compound detection if possible
    # For simplicity here, use regex pattern matching for compounds
    for i, token in enumerate(words):
        if re.match(pattern, token):
            word_index = i + 1  # Adjust index to skip [CLS]
            components = token.split("|")
            for component in components:
                # Each component is a part of the compound
                base_index = words.index(component.split("##")[-1])
                out[word_index, base_index + 1] = 1
                out[base_index + 1, word_index] = 1

    # Enable [CLS] and [SEP] to self-attend
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention pattern
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Compound-Compound Attention Linking", out