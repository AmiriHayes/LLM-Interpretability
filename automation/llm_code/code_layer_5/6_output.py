import numpy as np
import spacy

def adjunct_to_head_linking(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 5, Head 6 as an 'Adjunct-to-Head Linking' pattern.

    This function predicts attention from a word to the word that directly precedes and modifies it,
    including prepositions to verbs/nouns, adverbs to verbs, and other modifiers to their heads.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    try:
        # Load the spacy English model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model is not downloaded, download it
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)

    # Helper function to get BERT indices for a spaCy token
    def get_bert_indices(spacy_token_index):
        start_bert_idx = -1
        end_bert_idx = -1
        for i, word_id in enumerate(word_ids):
            if word_id == spacy_token_index:
                if start_bert_idx == -1:
                    start_bert_idx = i
                end_bert_idx = i
        if start_bert_idx != -1 and end_bert_idx != -1:
            return range(start_bert_idx, end_bert_idx + 1)
        return []
    
    # Iterate through spaCy tokens to find the adjunct-to-head pattern
    for token in doc:
        # Check if the token has a valid head (not root) and is a modifier
        if token.head != token and token.dep_ in ['prep', 'advmod', 'amod', 'acomp', 'dobj', 'pobj']:
            from_indices = get_bert_indices(token.i)
            to_indices = get_bert_indices(token.head.i)

            if from_indices and to_indices:
                # Pay high attention from the modifier to its head
                for from_idx in from_indices:
                    for to_idx in to_indices:
                        out[from_idx, to_idx] = 1.0
    
    # Add attention from sub-word tokens to the first token of the word
    for i in range(1, len_seq):
        if word_ids[i] is not None and word_ids[i] == word_ids[i-1]:
            out[i, i-1] = 1.0

    # Ensure all tokens have some self-attention to avoid zero rows
    np.fill_diagonal(out, out.diagonal() + 1.0)
    
    # Normalize matrix by row to represent a distribution
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predicted_matrix = out / row_sums

    return 'Adjunct-to-Head Linking Pattern', predicted_matrix