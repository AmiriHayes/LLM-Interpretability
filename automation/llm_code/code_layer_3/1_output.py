import numpy as np
import spacy
from typing import List, Tuple

def verb_centric_focus(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Predicts the attention pattern for Layer 3, Head 1, which is responsible
    for the 'Verb-Centric Focus Pattern'.

    This function identifies the main verb(s) of a sentence and models attention from
    modifiers and arguments to these verbs.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    tokens = tokenizer([sentence], return_tensors="pt")
    token_len = len(tokens.input_ids[0])
    
    predicted_matrix = np.zeros((token_len, token_len), dtype=float)

    # Use tokenizer's word_ids to map spaCy tokens to BERT tokens
    word_ids = tokens.word_ids(batch_index=0)
    
    # Map spaCy token index to a list of BERT token indices
    spacy_to_bert_map = {
        spacy_token_idx: [i for i, bert_word_id in enumerate(word_ids) if bert_word_id == spacy_token_idx]
        for spacy_token_idx, _ in enumerate(doc)
    }

    # Find the main verb and other relevant verbs (e.g., in clauses)
    main_verbs = [token for token in doc if token.pos_ == "VERB" or token.dep_ in ["ROOT", "advcl", "relcl", "pcomp", "acl"]]
    
    for verb in main_verbs:
        verb_bert_indices = spacy_to_bert_map.get(verb.i, [])

        # Find tokens that modify or are arguments of this verb
        for token in doc:
            # Check if a token is a subject, object, or modifier of the verb
            is_related = False
            if token.head == verb or (token.dep_ in ["nsubj", "dobj", "acomp", "advmod", "prep", "pobj", "ccomp"]):
                is_related = True
            
            # Additional check for subjects of participial phrases
            if verb.dep_ in ["advcl", "acl", "relcl"] and token.dep_ == "nsubj":
                is_related = True

            if is_related:
                related_bert_indices = spacy_to_bert_map.get(token.i, [])
                
                # Assign attention from the related token to the verb
                for from_idx in related_bert_indices:
                    for to_idx in verb_bert_indices:
                        if from_idx != to_idx:
                            predicted_matrix[from_idx, to_idx] += 0.5
                            
                # Assign some bidirectional attention for a stronger link
                for from_idx in verb_bert_indices:
                    for to_idx in related_bert_indices:
                        if from_idx != to_idx:
                            predicted_matrix[from_idx, to_idx] += 0.2

    # Assign self-attention for the CLS and EOS tokens
    predicted_matrix[0, 0] = 1.0
    if token_len > 1:
        predicted_matrix[token_len - 1, 0] = 1.0

    # Normalize the matrix rows so they sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Verb-Centric Focus Pattern', predicted_matrix