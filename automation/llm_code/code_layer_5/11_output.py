import numpy as np
import spacy
from typing import Tuple, List
from transformers import PreTrainedTokenizer

# Load a simple spaCy model for tokenization and POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def subject_predicate_attention(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 11.

    This function creates a rule-encoded attention matrix where tokens at the beginning of
    noun phrases (likely subjects) attend to the main verb or predicate of the sentence.
    It simulates a long-range dependency pattern that skips over intervening
    modifying phrases. It also includes attention from the first token of a sub-phrase
    to the token following it, when no long-range link is found.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    len_seq = len(token_ids)
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Use spaCy to find noun chunks and verbs
    doc = nlp(sentence)
    
    main_verb_idx = -1
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            main_verb_idx = token.i
            break

    # Map spaCy tokens to BERT tokens
    bert_tokens_and_ids = tokenizer.encode_plus(
        sentence, return_tensors="pt", return_token_type_ids=False, return_attention_mask=False
    )
    word_ids = bert_tokens_and_ids.word_ids(batch_index=0)
    
    # Heuristic for subject-to-verb attention
    # Find the index of the first token of the sentence subject
    subject_start_token_idx = -1
    if main_verb_idx != -1:
        for token in doc:
            # Check for a subject that precedes the main verb
            if (token.dep_ in ['nsubj', 'csubj'] and token.i < main_verb_idx):
                subject_start_token_idx = token.i
                break
    
    # Find the BERT indices for the subject and main verb
    if subject_start_token_idx != -1 and main_verb_idx != -1:
        bert_subject_indices = [i for i, word_id in enumerate(word_ids) if word_id == subject_start_token_idx]
        bert_verb_indices = [i for i, word_id in enumerate(word_ids) if word_id == main_verb_idx]
        
        if bert_subject_indices and bert_verb_indices:
            # First token of the subject attends to the first token of the main verb
            from_idx = bert_subject_indices[0]
            to_idx = bert_verb_indices[0]
            predicted_matrix[from_idx, to_idx] = 1.0

    # Fallback to general forward attention from phrase-beginnings
    # This captures the pattern where a token at the start of a modifying phrase
    # attends to a token later in that same phrase.
    # We will use a simple heuristic of a token attending to the next token
    # or a token at the beginning of a sub-phrase attending to a later token.
    
    # A simple, more general rule: tokens attend to the next token, particularly
    # across commas or to a preposition introducing a new clause.
    for i in range(1, len_seq - 1):
        # A simple forward attention to the next token
        predicted_matrix[i, i+1] = 0.5
        
        # Heuristic for linking phrases
        current_word_id = word_ids[i]
        next_word_id = word_ids[i+1]
        
        # If we are at the end of a word chunk, and the next word is a different word,
        # have the last token of the current word attend to the first token of the next word.
        if current_word_id != next_word_id and next_word_id is not None:
             # Look for a preposition or a conjunction that introduces a new part of the sentence
            if doc[next_word_id].pos_ in ['ADP', 'SCONJ', 'CCONJ']:
                predicted_matrix[i, i+1] = 1.0
            
    # Assign CLS and SEP tokens to have self-attention and some link to the sentence
    if len_seq > 1:
        predicted_matrix[0, 0] = 1.0  # CLS self-attention
        predicted_matrix[0, 1] = 1.0  # CLS to first word
    if len_seq > 2:
        predicted_matrix[-1, -1] = 1.0 # SEP self-attention
        predicted_matrix[-1, -2] = 1.0 # SEP to last word
        
    # Normalize each row to sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return 'Subject-to-Predicate/Modifier Attention', normalized_matrix