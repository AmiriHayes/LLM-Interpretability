import numpy as np
import spacy
from transformers import BertTokenizer

# Load a small spaCy model for part-of-speech and dependency parsing.
# You may need to run `python -m spacy download en_core_web_sm` once.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def question_negation_focus(sentence: str, tokenizer: BertTokenizer) -> tuple[str, np.ndarray]:
    """
    Predicts the attention matrix for Layer 2, Head 8, based on the hypothesis
    that it focuses on linking question/negation elements to the main verb or question mark.

    Args:
        sentence (str): The input sentence.
        tokenizer (BertTokenizer): The tokenizer for the model.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern
                                and the predicted attention matrix.
    """
    # Tokenize the sentence and get input IDs and word IDs
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    word_ids = tokens.word_ids(batch_index=0)
    seq_len = len(input_ids)
    
    # Initialize a zero matrix for attention weights
    predicted_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Use spaCy to parse the sentence for linguistic information
    doc = nlp(sentence)
    
    # Create a mapping from spaCy token index to a list of BERT token indices
    spacy_token_to_bert_indices = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in spacy_token_to_bert_indices:
                spacy_token_to_bert_indices[word_id] = []
            spacy_token_to_bert_indices[word_id].append(i)

    # Find the main verb(s) and question mark(s) in the sentence
    main_verbs_bert_indices = []
    question_mark_bert_indices = []
    
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "aux", "xcomp", "ccomp", "advcl", "conj"]:
            if token.i in spacy_token_to_bert_indices:
                main_verbs_bert_indices.extend(spacy_token_to_bert_indices[token.i])
        if token.text == "?":
            if token.i in spacy_token_to_bert_indices:
                question_mark_bert_indices.extend(spacy_token_to_bert_indices[token.i])

    # Apply the attention pattern
    for token in doc:
        # Get BERT token indices for the current spaCy token
        from_bert_indices = spacy_token_to_bert_indices.get(token.i, [])

        # Check for auxiliary verbs, interrogative words, and negation particles
        is_aux = token.pos_ == "AUX"
        is_interrogative = (token.pos_ in ["ADV", "PRON"] and token.dep_ in ["advmod", "nsubj"] and "?" in sentence) or \
                           (token.text.lower() in ["why", "what", "how", "when", "where", "who", "whom", "whose"])
        is_negation = token.dep_ == "neg" or token.text.lower() in ["not", "n't", "never"]

        if from_bert_indices and (is_aux or is_interrogative or is_negation):
            # Prioritize attention to the question mark if it exists
            if question_mark_bert_indices:
                for from_idx in from_bert_indices:
                    for to_idx in question_mark_bert_indices:
                        predicted_matrix[from_idx, to_idx] = 1.0
            # Otherwise, attend to main verbs
            elif main_verbs_bert_indices:
                for from_idx in from_bert_indices:
                    for to_idx in main_verbs_bert_indices:
                        predicted_matrix[from_idx, to_idx] = 1.0

    # Add self-attention for CLS and SEP tokens
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[seq_len - 1, seq_len - 1] = 1.0
    
    # Normalize each row to sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that have no attention targets
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Question/Negation Focus Pattern", normalized_matrix

# Example usage (for demonstration, not part of the required function)
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test_sentence_1 = "Why did the chicken cross the road?"
    pattern_name_1, pred_matrix_1 = question_negation_focus(test_sentence_1, tokenizer)
    print(f"Pattern Name: {pattern_name_1}")
    print("Predicted Attention Matrix Shape:", pred_matrix_1.shape)
    
    # Expected: attention from "Why", "did" to "?" and "cross"
    tokens_list_1 = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sentence_1, add_special_tokens=True))
    print(f"Tokens: {tokens_list_1}")
    
    # Find indices for 'why', 'did', 'cross', '?'
    why_idx = tokens_list_1.index('why') if 'why' in tokens_list_1 else -1
    did_idx = tokens_list_1.index('did') if 'did' in tokens_list_1 else -1
    cross_idx = tokens_list_1.index('cross') if 'cross' in tokens_list_1 else -1
    qm_idx = tokens_list_1.index('?') if '?' in tokens_list_1 else -1

    if why_idx != -1 and qm_idx != -1:
        print(f"Attention from 'why' (token {why_idx}) to '?' (token {qm_idx}): {pred_matrix_1[why_idx, qm_idx]:.4f}")
    if did_idx != -1 and qm_idx != -1:
        print(f"Attention from 'did' (token {did_idx}) to '?' (token {qm_idx}): {pred_matrix_1[did_idx, qm_idx]:.4f}")
    if did_idx != -1 and cross_idx != -1:
        print(f"Attention from 'did' (token {did_idx}) to 'cross' (token {cross_idx}): {pred_matrix_1[did_idx, cross_idx]:.4f}")

    test_sentence_2 = "He will not go."
    pattern_name_2, pred_matrix_2 = question_negation_focus(test_sentence_2, tokenizer)
    print(f"\nPattern Name: {pattern_name_2}")
    print("Predicted Attention Matrix Shape:", pred_matrix_2.shape)
    
    # Expected: attention from "not" to "go" and "will"
    tokens_list_2 = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sentence_2, add_special_tokens=True))
    print(f"Tokens: {tokens_list_2}")
    
    not_idx = tokens_list_2.index('not') if 'not' in tokens_list_2 else -1
    go_idx = tokens_list_2.index('go') if 'go' in tokens_list_2 else -1
    will_idx = tokens_list_2.index('will') if 'will' in tokens_list_2 else -1

    if not_idx != -1 and go_idx != -1:
        print(f"Attention from 'not' (token {not_idx}) to 'go' (token {go_idx}): {pred_matrix_2[not_idx, go_idx]:.4f}")
    if not_idx != -1 and will_idx != -1:
        print(f"Attention from 'not' (token {not_idx}) to 'will' (token {will_idx}): {pred_matrix_2[not_idx, will_idx]:.4f}")
