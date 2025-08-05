import spacy
import numpy as np

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def appositive_descriptor_pattern(sentence, tokenizer):
    """
    Hypothesizes an attention pattern for Layer 1, Head 8 based on
    connecting nouns to their appositive phrases or descriptive modifiers.

    Args:
        sentence (str): The input sentence.
        tokenizer: A BERT tokenizer instance.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get token length
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len))

    # Get word to token mapping
    word_ids = tokens.word_ids(batch_index=0)
    
    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Create a list of tuples to map spaCy token indices to BERT word_ids
    spacy_to_bert = [[] for _ in range(len(doc))]
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            spacy_to_bert[word_id].append(i)

    # Function to get the average BERT token index for a spaCy token
    def get_bert_index(spacy_token):
        bert_indices = spacy_to_bert[spacy_token.i]
        if not bert_indices:
            return None
        return int(np.mean(bert_indices))

    # Core logic to identify patterns
    for token in doc:
        # Connect nouns to their appositive modifiers
        if token.dep_ == "appos":
            head_bert_index = get_bert_index(token.head)
            appos_bert_index = get_bert_index(token)
            if head_bert_index is not None and appos_bert_index is not None:
                predicted_matrix[head_bert_index, appos_bert_index] = 1
                predicted_matrix[appos_bert_index, head_bert_index] = 1

        # Connect nouns to descriptive clauses (e.g., relative clauses)
        if token.dep_ == "relcl":
            relcl_head_bert_index = get_bert_index(token.head)
            relcl_bert_index = get_bert_index(token)
            if relcl_head_bert_index is not None and relcl_bert_index is not None:
                predicted_matrix[relcl_head_bert_index, relcl_bert_index] = 1
                predicted_matrix[relcl_bert_index, relcl_head_bert_index] = 1

        # Connect modifiers (adjectives, nouns) to the noun they describe
        if token.pos_ in ["ADJ", "NOUN"] and token.head.pos_ == "NOUN":
            modifier_bert_index = get_bert_index(token)
            noun_bert_index = get_bert_index(token.head)
            if modifier_bert_index is not None and noun_bert_index is not None:
                predicted_matrix[modifier_bert_index, noun_bert_index] = 1

        # Connect commas, conjunctions, and colons to the items they separate or list
        if token.text in [',', ':', 'and']:
            if token.n_lefts > 0 and token.n_rights > 0:
                left_child_bert_index = get_bert_index(token.left_edge)
                right_child_bert_index = get_bert_index(token.right_edge)
                if left_child_bert_index is not None and right_child_bert_index is not None:
                    predicted_matrix[left_child_bert_index, right_child_bert_index] = 1
                    predicted_matrix[right_child_bert_index, left_child_bert_index] = 1

    # Add self-attention for special tokens [CLS] and [SEP]
    predicted_matrix[0, 0] = 1
    predicted_matrix[token_len - 1, token_len - 1] = 1

    # Normalize the matrix by row to simulate attention weights
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = predicted_matrix / np.where(row_sums == 0, 1, row_sums)

    return 'Appositive and Descriptors Pattern', predicted_matrix