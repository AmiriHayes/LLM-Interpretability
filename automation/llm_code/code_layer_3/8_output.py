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

def noun_phrase_modifier(sentence, tokenizer):
    """
    Hypothesizes that this head links noun phrase modifiers to their head nouns.

    The function generates a predicted attention matrix by identifying head nouns
    and their associated modifiers (adjectives, other nouns, and words in a list).
    Attention is then projected from these modifiers back to the head noun.

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

    # Dictionary to map original words to their BERT token indices
    word_to_bert_idx = {}
    current_bert_idx = 1
    for word in doc:
        bert_tokens = tokenizer.tokenize(word.text)
        word_to_bert_idx[word.text.lower()] = list(range(current_bert_idx, current_bert_idx + len(bert_tokens)))
        current_bert_idx += len(bert_tokens)

    # Rule-based logic to find noun phrases and their modifiers
    for chunk in doc.noun_chunks:
        head_token = chunk.root
        head_bert_indices = word_to_bert_idx.get(head_token.text.lower(), [])

        if not head_bert_indices:
            continue

        for token in chunk:
            if token != head_token:
                modifier_bert_indices = word_to_bert_idx.get(token.text.lower(), [])
                
                # Direct attention from modifier tokens to the head noun tokens
                for mod_idx in modifier_bert_indices:
                    for head_idx in head_bert_indices:
                        predicted_matrix[mod_idx, head_idx] += 1.0

    # Punctuation-based list aggregation, e.g., "A, B, and C" attends to "C"
    # Or in the case of the provided data, a list aggregates to a preceding noun or a verb.
    # This part is a heuristic to capture the list aggregation pattern from the examples.
    for i, token in enumerate(doc):
        if token.pos_ in ['NOUN', 'PROPN']:
            if i > 0 and doc[i-1].text == ',' or i > 1 and doc[i-2].text == ',':
                # Heuristic: Find a likely head for the list.
                # Look backwards for a noun that introduced the list.
                # e.g., "bags: clothes, toiletries, a map" -> "clothes" attends to "bags"
                list_start = -1
                for j in range(i - 1, -1, -1):
                    if doc[j].text == ':' or (doc[j].pos_ == 'NOUN' and doc[j-1].text == ','):
                        list_start = j
                        break
                
                if list_start != -1:
                    list_head_bert_indices = word_to_bert_idx.get(doc[list_start].text.lower(), [])
                    if list_head_bert_indices:
                        for item_idx in word_to_bert_idx.get(token.text.lower(), []):
                            for head_idx in list_head_bert_indices:
                                predicted_matrix[item_idx, head_idx] += 1.0

    # Add self-attention and attention to CLS/SEP for all tokens
    for i in range(len_seq):
        predicted_matrix[i, i] += 1.0
    predicted_matrix[0, 0] += 1.0
    predicted_matrix[len_seq - 1, len_seq - 1] += 1.0
    
    # Normalize each row
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Noun Phrase Modifier Pattern', predicted_matrix