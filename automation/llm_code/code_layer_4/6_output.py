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

def predicate_to_subject_binding(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Predicate-to-Subject/Object Binding' attention pattern.

    This pattern is characterized by high attention from verbs, modal verbs,
    and participles to their subjects and/or objects. It is a back-pointing
    relationship from the predicate to the head of the noun phrase.

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

    # Initialize a low-attention matrix with a bias for self-attention on some special tokens
    predicted_matrix = np.full((seq_len, seq_len), 0.05)
    np.fill_diagonal(predicted_matrix, 0.1)

    # Use spaCy to find dependency relations
    doc = nlp(sentence)
    
    # Map spaCy token indices to tokenizer indices
    def get_token_indices(spacy_idx):
        return [i for i, wid in enumerate(word_ids) if wid == spacy_idx]
    
    # Iterate through tokens to find verbs and their dependents
    for token in doc:
        # Main Verbs and Auxiliaries
        if token.pos_ in ["VERB", "AUX"]:
            # Find subject (nsubj) and direct object (dobj)
            subject_indices = []
            object_indices = []
            
            # Find dependents in the sentence
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject_indices.extend(get_token_indices(child.i))
                if child.dep_ == "dobj":
                    object_indices.extend(get_token_indices(child.i))

            # Direct attention from the verb to its subjects and objects
            verb_indices = get_token_indices(token.i)
            for i in verb_indices:
                for j in subject_indices + object_indices:
                    predicted_matrix[i, j] = 0.6
        
        # Participles and related clauses (e.g., "splashing in puddles" in sentence 3)
        if token.pos_ == "VERB" and token.dep_ in ["csubj", "advcl", "relcl"]:
            head_verb_indices = get_token_indices(token.head.i)
            participle_indices = get_token_indices(token.i)
            
            # The participle/clause attends back to the main verb
            for i in participle_indices:
                for j in head_verb_indices:
                    predicted_matrix[i, j] = 0.5
        
        # Punctuation (like commas) often points back to the main verb in complex sentences
        if token.text == "," and token.head.pos_ in ["VERB", "AUX"]:
             comma_indices = get_token_indices(token.i)
             head_indices = get_token_indices(token.head.i)
             for i in comma_indices:
                 for j in head_indices:
                     predicted_matrix[i, j] = 0.4
    
    # Normalize rows to sum to 1.0
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predicted_matrix = predicted_matrix / row_sums

    return 'Predicate-to-Subject/Object Binding', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "She wondered, 'Will he ever understand the complexities of this intricate problem?'"
# pattern_name, matrix = predicate_to_subject_binding(sentence, tokenizer)
# print(pattern_name)
# print(matrix)