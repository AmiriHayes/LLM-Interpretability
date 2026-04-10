import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def abstract_entity_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy for linguistic annotations
    sentence_str = sentence[sentence.find("[CLS]")+5:sentence.find("[SEP]")].strip()
    doc = nlp(sentence_str)

    # Create a mapping from sentence indices to token indices
    spacy_to_tokenizer_map = {i: tok.i+1 for i, tok in enumerate(doc) if tok.text in toks["input_ids"][0]}

    # Potential keywords to look at, e.g., 'conceptual', 'idea', 'problem', 'mystery'
    # These are abstract entities in context usually attracting attention
    abstract_entities = {"hue", "complexities", "problem", "mysteries", "context", "symphony", "hope"}

    # Iterate over the sentence to identify abstract entities
    for tok in doc:
        if tok.text in abstract_entities and tok.i in spacy_to_tokenizer_map:
            token_index = spacy_to_tokenizer_map[tok.i]
            for child in tok.children:
                if child.dep_ in {"prep", "amod", "nmod", "acl"}:
                    # Check the map and give attention from the abstract entity to descriptors
                    if child.i in spacy_to_tokenizer_map:
                        child_index = spacy_to_tokenizer_map[child.i]
                        out[token_index, child_index] = 1
            # Abstract entity might attend to itself a little less
            out[token_index, token_index] += 0.5 

    # Ensure rows aren't all zeros
    for row in range(len_seq):
        if out[row].sum() == 0: out[row, -1] = 1.0

    # Normalize the out matrix by row
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Semantic Role Association with Abstract Entities", out