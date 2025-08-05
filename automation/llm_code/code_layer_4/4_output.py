import numpy as np
import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def svo_agreement(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'SVO Agreement' attention pattern.

    This pattern is characterized by high attention from the main subject
    and its modifiers to the main verb of the sentence. Attention is also
    seen from the final punctuation mark to the main verb, and from the
    subject to the object.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Tokenize and get word IDs
    toks = tokenizer([sentence], return_tensors="np", add_special_tokens=True)
    input_ids = toks["input_ids"][0]
    word_ids = toks.word_ids()
    seq_len = len(input_ids)
    
    # Initialize a low-attention matrix
    predicted_matrix = np.full((seq_len, seq_len), 0.05)
    np.fill_diagonal(predicted_matrix, 0.1)

    # Use spaCy to find the main subject and verb
    doc = nlp(sentence)
    
    main_verb_idx = -1
    subject_indices = []
    object_indices = []
    
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            main_verb_idx = token.i
            # Find the subject(s) and object(s) of this verb
            for child in token.children:
                if "subj" in child.dep_:
                    subject_indices.append(child.i)
                if "obj" in child.dep_:
                    object_indices.append(child.i)
            break
            
    if main_verb_idx != -1:
        # Get tokenizer indices for the main verb
        verb_tok_indices = [i for i, wid in enumerate(word_ids) if wid == main_verb_idx]
        
        # Link subject tokens to the verb
        for spacy_idx in subject_indices:
            subj_tok_indices = [i for i, wid in enumerate(word_ids) if wid == spacy_idx]
            for i in subj_tok_indices:
                for j in verb_tok_indices:
                    predicted_matrix[i, j] = 0.5
                    
        # Link object tokens to the verb
        for spacy_idx in object_indices:
            obj_tok_indices = [i for i, wid in enumerate(word_ids) if wid == spacy_idx]
            for i in obj_tok_indices:
                for j in verb_tok_indices:
                    predicted_matrix[i, j] = 0.3
                    
        # Link main verb back to subject and object
        for i in verb_tok_indices:
            for spacy_idx in subject_indices:
                subj_tok_indices = [j for j, wid in enumerate(word_ids) if wid == spacy_idx]
                for j in subj_tok_indices:
                    predicted_matrix[i, j] = 0.3
            for spacy_idx in object_indices:
                obj_tok_indices = [j for j, wid in enumerate(word_ids) if wid == spacy_idx]
                for j in obj_tok_indices:
                    predicted_matrix[i, j] = 0.2

    # Link final punctuation to the main verb if it exists
    if main_verb_idx != -1 and doc[-1].is_punct:
        final_punct_indices = [i for i, wid in enumerate(word_ids) if wid == len(doc)-1]
        verb_tok_indices = [i for i, wid in enumerate(word_ids) if wid == main_verb_idx]
        for i in final_punct_indices:
            for j in verb_tok_indices:
                predicted_matrix[i, j] = 0.4
    
    # Normalize rows to sum to 1.0
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'SVO Agreement Pattern', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The bustling city street, filled with cars, pedestrians, and street performers, was a symphony of sounds, sights, and smells."
# pattern_name, matrix = svo_agreement(sentence, tokenizer)
# print(pattern_name)
# print(matrix)