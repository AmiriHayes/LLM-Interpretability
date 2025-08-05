import numpy as np
import spacy

# Ensure spaCy model is downloaded. This block will only execute if the model is not found.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def predicate_structure_alignment(sentence, tokenizer):
    """
    Predicts attention patterns for Layer 6, Head 1, focusing on core predicate-argument
    structure and sentence-level attachment.

    This function identifies the main verb (root) and its subject, linking them
    bidirectionally. It also links the main verb to sentence boundaries ([CLS], [SEP])
    and handles attachment of introductory clauses/phrases.

    Args:
        sentence (str): The input sentence.
        tokenizer: A HuggingFace tokenizer (e.g., AutoTokenizer.from_pretrained("bert-base-uncased")).
                   Must support `__call__` with `return_offsets_mapping=True` to get word_ids.

    Returns:
        tuple: A tuple containing the pattern name (str) and the predicted attention matrix (numpy.ndarray).
               The matrix size is (token_len * token_len), where token_len includes [CLS] and [SEP].
    """
    # Tokenize the sentence and get word_ids to map subword tokens back to original words.
    # word_ids() maps subword token index to original word index (None for special tokens).
    encoded_input = tokenizer(sentence, return_offsets_mapping=True, return_attention_mask=True, return_token_type_ids=True, return_tensors="pt")
    input_ids = encoded_input['input_ids'][0]
    word_ids = encoded_input.word_ids()
    token_len = len(input_ids)

    predicted_matrix = np.zeros((token_len, token_len))

    # Rule 1: Self-attention for [CLS] and [SEP] tokens
    # These tokens often have a baseline self-attention.
    predicted_matrix[0, 0] = 1.0 # [CLS] token
    predicted_matrix[token_len - 1, token_len - 1] = 1.0 # [SEP] token

    # Parse the sentence with spaCy to get linguistic information (dependency tree, POS tags, etc.)
    doc = nlp(sentence)

    # Find the main verb (root of the dependency parse tree)
    main_verb_spacy_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            main_verb_spacy_token = token
            break

    if main_verb_spacy_token:
        main_verb_word_idx = main_verb_spacy_token.i
        # Get all subword token indices corresponding to the main verb
        main_verb_subword_indices = [i for i, w_id in enumerate(word_ids) if w_id == main_verb_word_idx]

        # Find the subject of the main verb (e.g., nominal subject, passive nominal subject)
        subject_spacy_token = None
        for child in main_verb_spacy_token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                subject_spacy_token = child
                break

        # Rule 2: Main Verb to Sentence Boundaries and Subject
        if main_verb_subword_indices:
            # Main verb attends to [CLS] (start of sentence) and [SEP] (end of sentence)
            for mv_idx in main_verb_subword_indices:
                predicted_matrix[mv_idx, 0] = 1.0 # To [CLS]
                predicted_matrix[mv_idx, token_len - 1] = 1.0 # To [SEP]

            # Main verb attends to its subject (if found)
            if subject_spacy_token:
                subject_word_idx = subject_spacy_token.i
                subject_subword_indices = [i for i, w_id in enumerate(word_ids) if w_id == subject_word_idx]
                for mv_idx in main_verb_subword_indices:
                    for subj_idx in subject_subword_indices:
                        predicted_matrix[mv_idx, subj_idx] = 1.0

        # Rule 3: Sentence Boundaries and Subject to Main Verb
        # [CLS] attends to main verb
        for mv_idx in main_verb_subword_indices:
            predicted_matrix[0, mv_idx] = 1.0
        # [SEP] attends to main verb
        for mv_idx in main_verb_subword_indices:
            predicted_matrix[token_len - 1, mv_idx] = 1.0

        # Subject attends to main verb (if found)
        if subject_spacy_token and subject_subword_indices:
            for subj_idx in subject_subword_indices:
                for mv_idx in main_verb_subword_indices:
                    predicted_matrix[subj_idx, mv_idx] = 1.0

        # Rule 4: Introductory Clause/Phrase Attachment (Comma or first content token to Main Verb)
        # Iterate through spaCy tokens to find potential introductory elements.
        # This rule focuses on linking tokens from introductory phrases/clauses to the main verb.
        for spacy_token in doc:
            # Link commas that precede the main verb (often marking introductory phrases) to the main verb.
            if spacy_token.text == ',' and spacy_token.i < main_verb_spacy_token.i:
                comma_subword_indices = [i for i, w_id in enumerate(word_ids) if w_id == spacy_token.i]
                for comma_idx in comma_subword_indices:
                    for mv_idx in main_verb_subword_indices:
                        predicted_matrix[comma_idx, mv_idx] = 1.0 # Comma to main verb

            # Link the first content word of the sentence to the main verb,
            # especially if it's not the main verb or its subject directly,
            # indicating it's part of an introductory construction.
            if spacy_token.i == 0 and \
               spacy_token != main_verb_spacy_token and \
               (subject_spacy_token is None or spacy_token != subject_spacy_token):
                first_word_subword_indices = [i for i, w_id in enumerate(word_ids) if w_id == 0]
                for first_idx in first_word_subword_indices:
                    for mv_idx in main_verb_subword_indices:
                        predicted_matrix[first_idx, mv_idx] = 1.0

    # Normalization: Ensure each row sums to 1.0.
    # For any row that is still all zeros (e.g., tokens not covered by explicit rules),
    # set self-attention to ensure a valid probability distribution.
    for i in range(token_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum
        else:
            # If a token has no specific attention rule applied, it defaults to self-attention.
            predicted_matrix[i, i] = 1.0

    return 'Sentence-Level Predication and Boundary Marking Pattern', predicted_matrix