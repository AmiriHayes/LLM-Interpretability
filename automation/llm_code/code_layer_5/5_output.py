import numpy as np
import spacy
from typing import Tuple, List
from transformers import PreTrainedTokenizer

nlp = spacy.load("en_core_web_sm")

def verb_to_subject_agreement(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 5 as 'Verb to Subject Agreement'.

    This function creates a rule-encoded attention matrix where the main verb of the
    sentence attends backward to its subject. This pattern is crucial for linking
    the action to the actor across potentially long distances.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    bert_tokens_and_ids = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="pt"
    )
    token_ids = bert_tokens_and_ids['input_ids'][0]
    word_ids = bert_tokens_and_ids.word_ids(batch_index=0)
    len_seq = len(token_ids)
    
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Use spaCy for dependency parsing
    doc = nlp(sentence)
    
    # Find the main verb and its subject(s)
    verb_indices = []
    subject_indices = []

    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            verb_indices.append(token.i)
            # Find the subject (nsubj) of this verb
            for child in token.children:
                if child.dep_ in ['nsubj', 'csubj']:
                    subject_indices.append(child.i)
    
    # Map spaCy indices to BERT indices and fill the matrix
    if verb_indices and subject_indices:
        main_verb_spacy_idx = verb_indices[0]
        subject_spacy_idx = subject_indices[0]

        # Get BERT indices for the verb and subject
        bert_verb_indices = [i for i, w_id in enumerate(word_ids) if w_id == main_verb_spacy_idx]
        bert_subject_indices = [i for i, w_id in enumerate(word_ids) if w_id == subject_spacy_idx]
        
        # The verb attends to the subject
        if bert_verb_indices and bert_subject_indices:
            from_idx = bert_verb_indices[0]
            to_idx = bert_subject_indices[0]
            predicted_matrix[from_idx, to_idx] = 1.0

    # Add self-attention for [CLS] and [SEP]
    if len_seq > 1:
        predicted_matrix[0, 0] = 1.0
    if len_seq > 2:
        predicted_matrix[-1, -1] = 1.0
    
    # Normalize rows to sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return 'Verb-to-Subject Agreement Pattern', normalized_matrix