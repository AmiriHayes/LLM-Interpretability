import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def coreference_punctuation_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 5, Head 2 as a 'Coreference Punctuation Pattern'.

    This function predicts attention from the final punctuation mark to the sentence's
    subject and other key words, and from tokens later in the sentence to the
    initial subject. It also predicts attention from the first token to the last.

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
        # Map spaCy tokens to BERT subword indices
        start_idx = bert_idx
        end_idx = start_idx
        for j in range(start_idx, len_seq - 1):
            if word_ids[j] == i:
                end_idx = j + 1
            else:
                break
        spacy_to_bert[i] = list(range(start_idx, end_idx))
        bert_idx = end_idx

    # Find sentence subject and final punctuation
    subject_tokens = []
    final_punct_tokens = []
    
    for token in doc:
        # Find the subject (often a noun or pronoun)
        if token.dep_ == "nsubj":
            subject_tokens.append(token)
        
    if doc[-1].is_punct:
        final_punct_tokens.append(doc[-1])

    # Coreference Pattern: Attention from final punctuation to the subject
    if final_punct_tokens and subject_tokens:
        from_indices = spacy_to_bert[final_punct_tokens[0].i]
        to_indices = []
        for subj_token in subject_tokens:
            if subj_token.i in spacy_to_bert:
                to_indices.extend(spacy_to_bert[subj_token.i])

        if from_indices and to_indices:
            for from_idx in from_indices:
                for to_idx in to_indices:
                    out[from_idx, to_idx] = 1.0

    # Bidirectional attention between the start/end of the sentence
    out[-1, 1] = 1.0  # From [SEP] to the first token
    out[1, -1] = 1.0  # From first token to [SEP]

    # Handle self-attention for special tokens
    out[0, 0] = 1.0  # [CLS] self-attention
    out[-1, -1] = 1.0  # [SEP] self-attention

    # Normalize the matrix by row (ensures a valid probability distribution)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    predicted_matrix = out / row_sums

    return 'Coreference Punctuation Pattern', predicted_matrix