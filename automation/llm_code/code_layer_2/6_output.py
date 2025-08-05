import numpy as np
import spacy

# Load the spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def appositive_attention(sentence, tokenizer):
    """
    Predicts an attention matrix for a head hypothesized to handle appositive
    and list-based attention.

    The pattern assumes attention flows from an introductory token (like a noun
    or a colon) to the tokens in the appositive or list that follows it.
    The function also gives high self-attention to key tokens within these lists.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer to use (e.g., from a Hugging Face model).

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted
               attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))

    # Get the spacy document
    doc = nlp(sentence)

    # Initialize lists to hold indices of list-like words and introductory words
    list_words_indices = []
    intro_words_indices = []

    # Find introductory punctuation (colons, dashes) and their lists
    for i, token in enumerate(doc):
        if token.text == ":" or (i > 0 and doc[i-1].text == ":" and token.pos_ == "NOUN"):
            intro_word = doc[i-1] if token.text != ":" else token
            intro_token_indices = [idx for idx, word_id in enumerate(word_ids) if word_id is not None and word_id == intro_word.i]
            if intro_token_indices:
                intro_words_indices.extend(intro_token_indices)

            # Find all words in the list that follows the colon or dash
            list_start_index = i
            for j in range(list_start_index, len(doc)):
                # Stop at a different sentence-ending punctuation or a new independent clause
                if doc[j].text in ['.', '?', '!', ';'] or doc[j].pos_ in ['VERB', 'CCONJ']:
                    break
                if doc[j].pos_ in ['NOUN', 'PROPN', 'ADJ', 'ADV', 'NUM']:
                    list_words_indices.extend([idx for idx, word_id in enumerate(word_ids) if word_id is not None and word_id == j])

    # Find appositive phrases
    for chunk in doc.noun_chunks:
        # A simple heuristic for appositives: a noun chunk followed by another noun chunk
        # separated by commas
        if chunk.root.dep_ == "appos":
            appositive_start_index = chunk.start
            appositive_end_index = chunk.end

            # Find the word the appositive is describing
            intro_word_root_index = chunk.root.head.i

            intro_token_indices = [idx for idx, word_id in enumerate(word_ids) if word_id is not None and word_id == intro_word_root_index]
            list_token_indices = [idx for idx, word_id in enumerate(word_ids) if word_id is not None and appositive_start_index <= word_id < appositive_end_index]
            
            if intro_token_indices and list_token_indices:
                intro_words_indices.extend(intro_token_indices)
                list_words_indices.extend(list_token_indices)


    # Apply attention rules to the matrix
    # Strong self-attention for key words in the lists/appositives
    for idx in list_words_indices:
        if idx < len_seq:
            out[idx, idx] = 1

    # Attention from the introductory words to the list words
    for from_idx in intro_words_indices:
        for to_idx in list_words_indices:
            if from_idx < len_seq and to_idx < len_seq:
                out[from_idx, to_idx] = 1

    # Assign self-attention for CLS and SEP tokens
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the matrix rows to sum to 1
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.where(row_sums > 0, out / row_sums, out)
    
    return 'Appositive/List Attention Pattern', out