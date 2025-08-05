import numpy as np
import spacy
from typing import List, Tuple
from transformers import PreTrainedTokenizer

def long_distance_linking(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 8 as a 'Long-Distance Linking' pattern.

    This function predicts attention from a subject noun/pronoun to its main verb or
    the head of its predicate, often skipping over intervening clauses or modifiers.
    It also links tokens within an appositive or parenthetical phrase to the
    main noun they modify.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    # Load a spaCy model for dependency parsing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback if model is not downloaded
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Tokenize and get word IDs
    toks = tokenizer([sentence], return_tensors="pt", add_special_tokens=True)
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)

    # Dictionary to map word ID to token indices
    word_to_token_map = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in word_to_token_map:
                word_to_token_map[word_id] = []
            word_to_token_map[word_id].append(i)

    # Loop through each token in the spaCy document
    for token in doc:
        # Get the start and end indices of the word in the BERT tokenization
        start_token_idx = word_to_token_map.get(token.i, [None])[0]
        
        if start_token_idx is None:
            continue
        
        # Check for subjects and their predicates
        if "subj" in token.dep_:
            # Find the root of the clause (the main verb)
            head = token.head
            head_idx = word_to_token_map.get(head.i, [None])[0]
            
            if head_idx is not None:
                # Subject attends to its main verb
                out[start_token_idx, head_idx] = 1.0

                # Also link to the end of the predicate for a "closure" effect
                for child in head.children:
                    if child.dep_ in ["acomp", "dobj", "pobj", "advcl", "ccomp"]:
                        child_idx = word_to_token_map.get(child.i, [None])[-1]
                        if child_idx is not None:
                            out[start_token_idx, child_idx] = 1.0
        
        # Check for appositive or parenthetical relations
        if token.dep_ == "appos":
            head = token.head
            head_idx = word_to_token_map.get(head.i, [None])[0]
            if head_idx is not None:
                # The appositive token attends to the noun it modifies
                out[start_token_idx, head_idx] = 1.0

    # Ensure special tokens have self-attention
    out[0, 0] = 1.0  # [CLS]
    out[-1, -1] = 1.0 # [SEP]

    # Normalize matrix by row to represent a distribution
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predicted_matrix = out / row_sums

    return 'Long-Distance Subject-Predicate Linking Pattern', predicted_matrix