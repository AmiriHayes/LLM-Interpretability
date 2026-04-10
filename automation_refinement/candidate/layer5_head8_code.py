from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import spacy

# Load the English NLP model in spaCy
nlp = spacy.load("en_core_web_sm")

def delayed_antecedent_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to analyze the sentence
    doc = nlp(sentence)

    # A dictionary to map spaCy tokens to BERT tokens
    spacy_to_bert = {}
    for i, tok in enumerate(doc):
        spacy_to_bert[i] = []
        encoded_word = tokenizer(tok.text, add_special_tokens=False).input_ids
        decoded_tokens = [tokenizer.decode([id]) for id in encoded_word]
        # Match the decoded tokens to their positions in BERT input IDs
        for j in range(len_seq):
            if tokenizer.decode([toks.input_ids[0][j]]) in decoded_tokens:
                spacy_to_bert[i].append(j)

    # Scan through each token in the sentence
    for token in doc:
        if token.dep_ == "nsubj" or token.dep_ == "dobj":  # For example, looking at subjects and direct objects
            subj_antecedent = token.i  # Current token
            antecedent_head = token.head.i  # Word to which it's linked
            if antecedent_head != subj_antecedent:
                antecedent_positions = spacy_to_bert.get(subj_antecedent, [])
                head_positions = spacy_to_bert.get(antecedent_head, [])
                # Create the attention links based on the spaCy analysis
                for pos in antecedent_positions:
                    for h_pos in head_positions:
                        out[h_pos, pos] = 1
                # Add more delayed recognition patterns if needed

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Delayed Antecedent Recognition", out