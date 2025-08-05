import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def prepositional_phrase_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 5, Head 1 as a 'Prepositional Phrase Pattern'.

    This function predicts that the head attends from a preposition to the head noun
    of the preceding noun phrase. It uses spaCy's dependency parsing to identify
    prepositions and their governing nouns.

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
        # Handle subword tokens from BERT tokenizer
        for j in range(bert_idx, len_seq - 1):
            if word_ids[j] == i:
                if i not in spacy_to_bert:
                    spacy_to_bert[i] = []
                spacy_to_bert[i].append(j)
                bert_idx = j + 1
            else:
                break
    
    # Identify prepositional phrase dependencies
    for token in doc:
        if token.dep_ == "prep":  # Check for prepositions
            # The head of the preposition is often the noun it's modifying
            head_noun = token.head
            
            # Find the BERT indices for the preposition and the head noun
            if token.i in spacy_to_bert and head_noun.i in spacy_to_bert:
                from_indices = spacy_to_bert[token.i]
                to_indices = spacy_to_bert[head_noun.i]
                
                # For each subword of the preposition, attend to each subword of the head noun
                for from_idx in from_indices:
                    for to_idx in to_indices:
                        out[from_idx, to_idx] = 1.0

    # Handle self-attention for special tokens
    out[0, 0] = 1.0  # [CLS] self-attention
    out[-1, -1] = 1.0  # [SEP] self-attention

    # Normalize the matrix by row
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    predicted_matrix = out / row_sums

    return 'Prepositional Phrase Pattern', predicted_matrix