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

def appositive_subject_linkage(sentence, tokenizer):
    """
    Hypothesizes that this head links appositive phrases and clauses to the
    main subject or noun they modify.

    The function generates a predicted attention matrix by identifying a head noun and
    any modifying clauses, often set off by commas. Attention is then
    projected from the tokens within the modifying clause back to the head noun.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    len_seq = len(tokenized_sentence)
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    doc = nlp(sentence)
    
    # Initialize a list of (start_idx, end_idx) for appositive-like phrases
    appositive_spans = []
    
    # Simple rule-based search for appositives and parenthetical phrases
    # using commas as delimiters.
    comma_indices = [i for i, tok in enumerate(doc) if tok.text == ',']
    
    # If there are at least two commas, we can look for a modifying phrase
    if len(comma_indices) >= 2:
        for i in range(len(comma_indices) - 1):
            start_idx = comma_indices[i]
            end_idx = comma_indices[i+1]
            
            # This is a heuristic: check if the phrase is a non-essential
            # modifier. For example, "The house, standing on the hill, was old."
            # The part "standing on the hill" is an appositive.
            # We assume a pattern where a noun is followed by a comma, then the appositive, then another comma.
            if start_idx > 0:
                head_noun = doc[start_idx-1]
                if head_noun.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                    appositive_spans.append((head_noun, start_idx+1, end_idx))

    # Process attention for each identified appositive phrase
    for head_noun_spacy, appos_start_spacy, appos_end_spacy in appositive_spans:
        # Find the token indices in the BERT tokenization
        head_noun_bert_idx = -1
        current_idx = 0
        for i, bert_token in enumerate(tokenized_sentence):
            if bert_token.lower().startswith(head_noun_spacy.text.lower()):
                head_noun_bert_idx = i
                break

        appos_start_bert_idx = -1
        appos_end_bert_idx = -1
        for i, bert_token in enumerate(tokenized_sentence):
            if i > head_noun_bert_idx and bert_token.lower().startswith(doc[appos_start_spacy].text.lower()):
                 appos_start_bert_idx = i
                 break
        for i, bert_token in reversed(list(enumerate(tokenized_sentence))):
            if i < len_seq-1 and bert_token.lower().startswith(doc[appos_end_spacy].text.lower()):
                 appos_end_bert_idx = i
                 break

        if head_noun_bert_idx != -1 and appos_start_bert_idx != -1 and appos_end_bert_idx != -1:
            for i in range(appos_start_bert_idx, appos_end_bert_idx + 1):
                # Tokens in the appositive phrase attend to the head noun
                predicted_matrix[i, head_noun_bert_idx] += 1.0

    # For all tokens, add self-attention.
    for i in range(len_seq):
        predicted_matrix[i, i] += 1.0
        
    # The first token ([CLS]) and last token ([SEP]) also attend to each other
    predicted_matrix[0, len_seq - 1] += 1.0
    predicted_matrix[len_seq - 1, 0] += 1.0
    
    # Normalize each row
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Appositive-Subject Linkage Pattern', predicted_matrix