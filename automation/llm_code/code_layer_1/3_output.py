import numpy as np
import spacy
from typing import List

def coreference_resolution_for_lists(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Coreference Resolution for Lists' pattern for Layer 1, Head 3.

    This function predicts attention from each word in a list-like structure back to a
    centralized 'head' token, which can be the noun being described, the preceding
    punctuation, or other list items. The pattern is encoded into a matrix
    where tokens that are part of a list attend to a designated 'head' token.
    The function uses spaCy to identify nouns and list-like structures to
    generalize this pattern. It returns the name of the pattern and the predicted
    attention matrix.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer to process the sentence.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    tokenized_input = tokenizer([sentence], return_tensors="pt")
    input_ids = tokenized_input['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_ids = tokenized_input.word_ids()
    len_seq = len(input_ids)
    
    predicted_matrix = np.zeros((len_seq, len_seq))
    
    # Identify list items and potential head tokens using spaCy
    list_head_map = {}
    
    # Iterate through the spaCy tokens to find lists
    for token in doc:
        # Check for list-like structures, e.g., commas separating nouns or adjectives
        if token.pos_ in ["NOUN", "ADJ"] and (token.head.pos_ == "NOUN" or token.head.text in [",", ":"]):
            head_token = token.head
            # Handle cases where the head token is punctuation, and find the actual noun
            if head_token.text in [",", ":"] and head_token.head:
                head_token = head_token.head
            
            # Map all tokens belonging to the list to the main head token
            for t in doc:
                if t.head == token or t == token:
                    list_head_map[t.i] = head_token.i
    
    # Encode the pattern into the predicted matrix
    for i in range(len_seq):
        from_word_id = word_ids[i]
        
        # Self-attention for special tokens like [CLS] and [SEP]
        if from_word_id is None:
            predicted_matrix[i, i] = 1.0
            continue
            
        # Find the main head token for the current token if it's part of a list
        if from_word_id in list_head_map:
            head_word_id = list_head_map[from_word_id]
            
            # Find the tokenizer indices for the head token
            head_indices = [j for j, w_id in enumerate(word_ids) if w_id == head_word_id]
            
            if head_indices:
                # Distribute attention from the current token to all tokens of the head
                attention_value = 1.0 / len(head_indices)
                for head_idx in head_indices:
                    predicted_matrix[i, head_idx] = attention_value
        else:
            # For non-list tokens, assign self-attention
            predicted_matrix[i, i] = 1.0

    # Normalizing each row to sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Coreference Resolution for Lists', predicted_matrix

# Example usage (commented out as per instructions, but for demonstration)
# if __name__ == '__main__':
#     from transformers import BertTokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     sentence = "The bustling city street, filled with cars, pedestrians, and street performers, was a symphony of sounds, sights, and smells."
#     pattern_name, pred_matrix = coreference_resolution_for_lists(sentence, tokenizer)
#     print(f"Pattern Name: {pattern_name}")
#     print(f"Predicted Matrix Shape: {pred_matrix.shape}")
#     print("Predicted Matrix (partial):")
#     print(pred_matrix[0:5, 0:5])