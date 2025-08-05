import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def modifier_head_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 5, Head 3 as a 'Modifier-Head Pattern'.

    This function predicts attention from modifying tokens (adverbs, adjectives)
    to the head word they modify. It also captures attention from conjunctions
    and punctuation to their preceding head words, and within prepositional phrases.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    
    spacy_to_bert = {}
    bert_idx = 1
    for i, spacy_token in enumerate(doc):
        start_idx = bert_idx
        end_idx = start_idx
        for j in range(start_idx, len_seq - 1):
            if word_ids[j] == i:
                end_idx = j + 1
            else:
                break
        spacy_to_bert[i] = list(range(start_idx, end_idx))
        bert_idx = end_idx

    # Pattern: Attention from a modifier to its head word.
    for token in doc:
        if token.head != token and token.i in spacy_to_bert and token.head.i in spacy_to_bert:
            
            # Check for a variety of dependency relations that signify modification
            # e.g., advmod (adverbial modifier), amod (adjectival modifier),
            # npadvmod (noun phrase as adverbial modifier)
            modifier_relations = ["advmod", "amod", "prep", "acomp", "advcl", "dobj", "pobj", "nsubj"]
            
            # Also consider conjunctions and their related words
            if token.dep_ in modifier_relations or token.dep_ == "cc" or token.dep_ == "conj" or token.is_punct:
                from_indices = spacy_to_bert[token.i]
                to_indices = spacy_to_bert[token.head.i]

                for from_idx in from_indices:
                    for to_idx in to_indices:
                        out[from_idx, to_idx] = 1.0

    # Punctuation to the word it follows
    for i in range(1, len(doc)):
        if doc[i].is_punct and doc[i].i in spacy_to_bert and doc[i-1].i in spacy_to_bert:
            from_indices = spacy_to_bert[doc[i].i]
            to_indices = spacy_to_bert[doc[i-1].i]
            for from_idx in from_indices:
                for to_idx in to_indices:
                    out[from_idx, to_idx] = 1.0

    # Handling special tokens
    out[0, 0] = 1.0  # [CLS] self-attention
    out[-1, -1] = 1.0 # [SEP] self-attention

    # Normalize the matrix by row
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with no attention
    predicted_matrix = out / row_sums

    return 'Modifier-Head Pattern', predicted_matrix