import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

#0
def pos_co_reference(sentence: str, tokenizer) -> (str, np.ndarray):
    """
    Predicts attention patterns based on a hypothesis of Adjective-Noun and
    Noun-Noun Co-Reference.

    The function identifies adjectives and nouns using spaCy and then encodes
    a predicted attention matrix where adjectives attend to their modified nouns,
    and nouns in a series (separated by commas or conjunctions) attend to each other.
    
    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer to be used for the sentence.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern and the
                                predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks['input_ids'][0]
    word_ids = toks.word_ids()
    
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len))

    doc = nlp(sentence)
    
    # Map spaCy tokens to BERT word_ids
    spacy_word_id_map = {}
    for i, token in enumerate(doc):
        # find the first word_id that matches the spaCy token's start and end char indices
        start_char = token.idx
        end_char = token.idx + len(token.text)
        
        for bert_id, word_id in enumerate(word_ids):
            if word_id is not None and word_id == i:
                if (tokenizer.decode(input_ids[bert_id:bert_id+1]).strip() == token.text or
                    tokenizer.decode(input_ids[bert_id:bert_id+1]).strip() in token.text or
                    token.text in tokenizer.decode(input_ids[bert_id:bert_id+1]).strip()):
                    if i not in spacy_word_id_map:
                         spacy_word_id_map[i] = []
                    spacy_word_id_map[i].append(bert_id)

    # Adjective-Noun Co-reference
    for token in doc:
        if token.pos_ == "ADJ":
            for child in token.children:
                if child.pos_ == "NOUN":
                    # Adjective attends to the noun it modifies
                    if token.i in spacy_word_id_map and child.i in spacy_word_id_map:
                        adj_indices = spacy_word_id_map[token.i]
                        noun_indices = spacy_word_id_map[child.i]
                        for adj_idx in adj_indices:
                            for noun_idx in noun_indices:
                                predicted_matrix[adj_idx, noun_idx] = 1

    # Noun-Noun Co-reference (for lists)
    nouns_in_list = []
    for token in doc:
        if token.pos_ == "NOUN":
            # Check for a preceding comma or conjunction, indicating a list
            if any(c.text in [",", "and", "or"] for c in token.children) or \
               any(c.text in [",", "and", "or"] for c in token.head.children):
                nouns_in_list.append(token)
    
    for i in range(len(nouns_in_list)):
        for j in range(len(nouns_in_list)):
            if i != j:
                # Nouns in a list attend to each other
                if nouns_in_list[i].i in spacy_word_id_map and nouns_in_list[j].i in spacy_word_id_map:
                    from_indices = spacy_word_id_map[nouns_in_list[i].i]
                    to_indices = spacy_word_id_map[nouns_in_list[j].i]
                    for from_idx in from_indices:
                        for to_idx in to_indices:
                            predicted_matrix[from_idx, to_idx] = 1

    # Add attention for special tokens [CLS] and [SEP] and self-attention for all tokens
    for i in range(token_len):
        predicted_matrix[i, i] = 1  # All tokens have self-attention
    
    predicted_matrix[0, 0] = 1  # [CLS] attends to itself
    predicted_matrix[-1, 0] = 1 # [SEP] attends to [CLS]

    # Normalize each row to sum to 1 to simulate uniform attention
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)
    
    return 'Adjective-Noun and Noun-Noun Co-reference Pattern', predicted_matrix

import numpy as np
import torch
import spacy

#1
def conjunctive_alignment(sentence, tokenizer):
    """
    Hypothesizes a 'Conjunctive Alignment Pattern' for a given sentence.
    This pattern links list-separating tokens (like commas and colons)
    to the conjunctions that connect the list items, or to the beginning
    of the list itself.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Use spacy for linguistic analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model isn't downloaded, download it
        print("Downloading spacy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(sentence)
    
    # Tokenize and get word IDs for alignment
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    word_ids = tokens.word_ids(batch_index=0)
    
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len), dtype=np.float32)

    # Dictionary to map spacy token index to tokenizer token indices
    spacy_to_tokenizer = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in spacy_to_tokenizer:
                spacy_to_tokenizer[word_id] = []
            spacy_to_tokenizer[word_id].append(i)

    # Find conjunctions and punctuation that act as list separators
    list_separators = []
    conjunctions = []
    
    for i, token in enumerate(doc):
        # Spacy token index
        spacy_idx = token.i
        # Tokenizer indices
        tokenizer_indices = spacy_to_tokenizer.get(spacy_idx, [])

        if token.text == "and" or token.text == "&":
            conjunctions.extend(tokenizer_indices)
        elif token.text in [',', ':']:
            list_separators.extend(tokenizer_indices)

    # Rule: For each list separator, attend to the conjunctions
    for from_idx in list_separators:
        for to_idx in conjunctions:
            # We add a weight. A value of 1 here means this is a strong connection.
            predicted_matrix[from_idx, to_idx] = 1.0

    # Additional rule to capture cases like "a, b, c"
    # A list separator can attend to other list separators that precede it
    for i, from_idx in enumerate(list_separators):
        if i > 0:
            for j in range(i):
                to_idx = list_separators[j]
                predicted_matrix[from_idx, to_idx] = 0.5  # A slightly weaker connection

    # Add a self-attention for CLS token and other tokens
    for i in range(token_len):
        predicted_matrix[i, i] = 1.0
    
    # Normalize the matrix rows so they sum to 1, mirroring attention weights
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows with no attention
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Conjunctive Alignment Pattern', predicted_matrix

import numpy as np
import spacy
from transformers import BertTokenizer

# Load spaCy for tokenization and linguistic features
nlp = spacy.load("en_core_web_sm")

def subword_reassembly(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 0, Head 2, based on the subword reassembly pattern.
    
    This function creates a predicted attention matrix where tokens attend to their preceding subword
    tokens, thereby "reassembling" the full word. The attention is a directed, local, and
    high-confidence signal for tokens that are continuations of a previous word.

    Args:
        sentence (str): The input sentence to analyze.
        tokenizer (BertTokenizer): A pre-trained tokenizer like BertTokenizer.

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted attention matrix.
               The matrix size is (token_len * token_len).
    """

    # Tokenize the sentence to get token IDs and word IDs
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Get the word IDs, which map tokens back to original words.
    # Note: `word_ids` requires the tokenizer to be instantiated with `add_special_tokens=True`
    # and a version that supports this feature.
    word_ids = encoded.word_ids()

    # Initialize the predicted attention matrix
    len_seq = len(input_ids)
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    # Use spaCy for more robust word-level analysis if needed, though word_ids handles most cases
    # doc = nlp(sentence)
    
    # Iterate through the tokens to identify subword relationships
    for i in range(1, len_seq):
        from_token_id = i
        to_token_id = i - 1

        # Check for subword tokens. BERT's tokenizer typically uses '##' for continuations.
        if tokens[from_token_id].startswith('##'):
            # This is a subword token, so it should attend to the previous token.
            # This is a simple but effective rule for this head.
            predicted_matrix[from_token_id, to_token_id] = 1.0
        
        # A more robust check using word_ids
        elif word_ids[from_token_id] is not None and word_ids[to_token_id] is not None:
            if word_ids[from_token_id] == word_ids[to_token_id]:
                # If both tokens belong to the same original word, the latter token
                # attends to the former. This covers multi-part subword tokens.
                predicted_matrix[from_token_id, to_token_id] = 1.0

    # Handle special tokens for self-attention.
    # The [CLS] token and the final token typically attend to themselves.
    # The first row of the attention matrix (from [CLS]) often shows self-attention or
    # is unassigned in this pattern, so we'll assign self-attention.
    predicted_matrix[0, 0] = 1.0
    
    # Normalize each row to sum to 1. This mimics the softmax operation in attention mechanisms.
    # We add a small value to avoid division by zero if a row is all zeros.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    predicted_matrix = predicted_matrix / row_sums

    return 'Subword Reassembly Pattern', predicted_matrix

# Example usage:
# sentence = "The old, creaky house, standing on the hill, seemed to whisper secrets."
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# pattern_name, predicted_matrix = subword_reassembly(sentence, tokenizer)
# print(f"Pattern: {pattern_name}")
# print("Predicted Attention Matrix:")
# print(predicted_matrix)

import numpy as np
import spacy

def modifier_head_alignment(sentence, tokenizer):
    """
    Hypothesizes a Modifier-Head Alignment Pattern for Layer 0, Head 3 of a BERT model.

    This function predicts an attention pattern where head tokens (nouns, verbs) attend
    to their preceding modifiers (adjectives, adverbs, determiners). It uses a rule-based
    approach with spaCy to identify these linguistic relationships and encode them
    into a attention matrix.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer used to tokenize the sentence.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    
    # Load spaCy model for linguistic parsing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download the model if it's not present
        print("Downloading spaCy 'en_core_web_sm' model. This may take a moment...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Tokenize the sentence and get word IDs for alignment
    tokenized_input = tokenizer(sentence, return_tensors="pt")
    input_ids = tokenized_input.input_ids[0]
    word_ids = tokenized_input.word_ids()
    token_len = len(input_ids)
    
    # Initialize the attention matrix
    predicted_matrix = np.zeros((token_len, token_len))

    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Create a mapping from spaCy tokens to BERT token indices
    token_to_idx = {doc_token: [] for doc_token in doc}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            token_to_idx[doc[word_id]].append(i)
    
    # Encode the Modifier-Head Alignment pattern
    for token in doc:
        # Check for direct modifier-head relationships
        # A token is a head if other tokens are its children
        if len(list(token.children)) > 0:
            for child in token.children:
                # Check if the child is a modifier (adjective, adverb, determiner)
                # and comes before the head token in the sentence
                if child.pos_ in ["ADJ", "ADV", "DET"] and child.i < token.i:
                    # Get the BERT token indices for the modifier and the head
                    head_indices = token_to_idx.get(token, [])
                    modifier_indices = token_to_idx.get(child, [])
                    
                    # If the head has a preceding modifier, it will attend to it
                    for head_idx in head_indices:
                        for mod_idx in modifier_indices:
                            predicted_matrix[head_idx, mod_idx] = 1

    # Apply uniform attention to the start token (CLS) and end token (SEP) as is common in BERT
    predicted_matrix[0, 0] = 1  # CLS self-attention
    predicted_matrix[-1, -1] = 1 # SEP self-attention

    # Normalize each row to sum to 1 to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows with no attention
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Modifier-Head Alignment Pattern', predicted_matrix

# Example of how to use the function (requires transformers library)
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The intricate details of the ancient tapestry told a fascinating story."
# pattern_name, matrix = modifier_head_alignment(sentence, tokenizer)
# print(f"Pattern Name: {pattern_name}")
# print("Predicted Attention Matrix:")
# print(matrix)

import numpy as np
import spacy

def descriptive_advancement_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 0, Head 4, which seems to follow
    a 'Descriptive Advancement Pattern'. This pattern links a main noun and its 
    related descriptive elements (adjectives, participles, etc.) to a subsequent 
    verb or noun that furthers the description or action related to the initial noun.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern and the 
                                predicted attention matrix.
    """
    # Load spacy model for linguistic analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback for environments where the model is not downloaded
        print("Downloading spaCy model 'en_core_web_sm'. This will happen only once.")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Tokenize and get word IDs for aligning with spaCy doc
    tokenized_input = tokenizer(sentence, return_tensors="pt")
    input_ids = tokenized_input['input_ids'][0]
    word_ids = tokenized_input.word_ids(batch_index=0)
    
    seq_len = len(input_ids)
    predicted_matrix = np.zeros((seq_len, seq_len))

    # Process sentence with spaCy
    doc = nlp(sentence)
    
    # Map spaCy token indices to BERT token indices
    spacy_to_bert = [[] for _ in range(len(doc))]
    bert_to_spacy = {}
    current_spacy_idx = 0
    for i, bert_word_id in enumerate(word_ids):
        if bert_word_id is not None and bert_word_id < len(doc):
            if i > 0 and bert_word_id != word_ids[i-1]:
                current_spacy_idx = bert_word_id
            spacy_to_bert[current_spacy_idx].append(i)
            bert_to_spacy[i] = current_spacy_idx
    
    # --- The Core Logic for the Descriptive Advancement Pattern ---
    
    # 1. Identify key nouns and their descriptive elements and related verbs
    main_nouns = [i for i, token in enumerate(doc) if token.pos_ in ["NOUN", "PROPN"]]
    descriptive_elements = [i for i, token in enumerate(doc) if token.pos_ in ["ADJ", "VERB"] and token.dep_ in ["amod", "acl", "advcl", "relcl"]]
    advancing_verbs_or_nouns = [i for i, token in enumerate(doc) if token.pos_ in ["VERB", "NOUN", "PROPN"]]

    # 2. Loop through spacy tokens and apply the rule
    for from_spacy_idx in main_nouns + descriptive_elements:
        # Find the next verb or noun that could 'advance' the description
        for to_spacy_idx in advancing_verbs_or_nouns:
            if to_spacy_idx > from_spacy_idx: # Attention flows forward
                # Check for a conceptual link (e.g., dependency, or just proximity)
                # This is a simplified heuristic. Real-world patterns are complex.
                # Here, we'll link descriptive words/main nouns to any subsequent verbs/nouns
                # This mirrors the observed pattern of linking a subject to its action, 
                # and its modifiers to those same actions/nouns.
                
                # Get the BERT token indices for the spacy tokens
                from_bert_indices = spacy_to_bert[from_spacy_idx]
                to_bert_indices = spacy_to_bert[to_spacy_idx]

                if from_bert_indices and to_bert_indices:
                    for from_idx in from_bert_indices:
                        for to_idx in to_bert_indices:
                            # Add weight to the matrix. A value of 1 represents a link.
                            # We'll normalize later.
                            predicted_matrix[from_idx, to_idx] = 1

    # 3. Handle special tokens: [CLS] and [SEP]
    # In BERT, CLS and SEP often have self-attention and/or attention to the start of the sentence
    # This is a common pattern for many heads.
    predicted_matrix[0, 0] = 1 # [CLS] self-attention
    predicted_matrix[-1, 0] = 1 # [SEP] attention to [CLS]
    
    # Normalize the matrix by row to simulate attention probabilities
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that have no attention
    row_sums[row_sums == 0] = 1
    normalized_matrix = predicted_matrix / row_sums

    return "Descriptive Advancement Pattern", normalized_matrix

import numpy as np
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def adjective_noun_alignment(sentence, tokenizer):
    """
    Predicts an attention matrix for the Adjective-to-Noun Alignment Pattern.

    This function identifies adjectives and nouns in a sentence and creates a 
    predicted attention matrix where adjectives attend to the closest noun.
    The pattern also includes bidirectional attention and self-attention for
    special tokens.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A tokenizer object with a `tokenize` method and 
                 `convert_tokens_to_ids` method.

    Returns:
    - tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    
    # Tokenize the sentence and get the sequence length
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_len = len(tokens)
    
    # Initialize the predicted matrix with zeros
    predicted_matrix = np.zeros((token_len, token_len))

    # Process the sentence with SpaCy to get part-of-speech tags
    doc = nlp(sentence)
    
    # Align SpaCy tokens with tokenizer tokens
    # Note: This is a simplified alignment. A more robust implementation would
    # handle subword tokens carefully.
    spacy_token_map = {token.text: token.pos_ for token in doc}
    
    # Get the part-of-speech for each BERT token
    # This loop is a simple heuristic; subword tokens (like '##ing') will
    # inherit the POS of the full word they belong to.
    token_pos = []
    for token in tokens:
        if token.startswith('##'):
            # Simple assumption: subword tokens get the same POS as the full word
            # This is not a perfect approach but works for a heuristic.
            pass
        elif token in spacy_token_map:
            token_pos.append(spacy_token_map[token])
        else:
            token_pos.append(None)
    
    # Find all adjectives and nouns
    adjective_indices = [i for i, pos in enumerate(token_pos) if pos == 'ADJ']
    noun_indices = [i for i, pos in enumerate(token_pos) if pos == 'NOUN']

    # Encode the Adjective-to-Noun Alignment pattern
    for adj_idx in adjective_indices:
        # Find the closest noun to the adjective
        closest_noun_idx = -1
        min_distance = float('inf')
        
        for noun_idx in noun_indices:
            distance = abs(adj_idx - noun_idx)
            if distance < min_distance:
                min_distance = distance
                closest_noun_idx = noun_idx
        
        if closest_noun_idx != -1:
            # High attention from adjective to its closest noun
            predicted_matrix[adj_idx, closest_noun_idx] = 1.0
            # Also, some bidirectional attention from noun to adjective
            predicted_matrix[closest_noun_idx, adj_idx] = 0.5

    # Add self-attention for special tokens [CLS] and [SEP]
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[token_len - 1, token_len - 1] = 1.0
    
    # Normalize each row to sum to 1, simulating attention weights
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = predicted_matrix / np.where(row_sums == 0, 1, row_sums)

    return 'Adjective-to-Noun Alignment Pattern', predicted_matrix

import numpy as np
import spacy
from transformers import BertTokenizer
from nltk.corpus import stopwords
import re

# Load a small spaCy model for part-of-speech tagging and lemmatization.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_predicted_attention_pattern(sentence, tokenizer):
    """
    Predicts the attention pattern for BERT's Layer 0, Head 6 based on
    the 'Sentence Theme Summarization' hypothesis.

    The function's logic is to identify a central 'theme' token (often
    the first significant noun, adjective, or gerund) and then have all
    other tokens attend to it. Additionally, all tokens maintain a strong
    self-attention link. This models the observed pattern of a single
    word becoming a hub of attention for the entire sentence.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    
    # Tokenize the sentence and get the BERT token IDs and word IDs.
    # The word_ids help map sub-word tokens back to the original words.
    tokenized_sentence = tokenizer(
        [sentence],
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    tokens = tokenized_sentence['input_ids'][0]
    token_len = len(tokens)
    
    # Initialize the attention matrix with zeros.
    predicted_matrix = np.zeros((token_len, token_len))
    
    # Use spaCy to parse the sentence and find the 'theme' word.
    # We'll use the first non-stopword, non-punctuation noun, adjective,
    # or gerund as a heuristic for the theme.
    doc = nlp(sentence)
    theme_word_text = None
    
    # Heuristic for finding the theme word:
    # 1. Look for the first adjective (ADJ) or noun (NOUN).
    # 2. If not found, look for a gerund (VERB with tag 'VBG').
    # 3. If still not found, use the first non-stopword token.
    for token in doc:
        # Exclude special tokens and stopwords.
        if token.is_punct or token.is_stop or token.text in ['[CLS]', '[SEP]']:
            continue
        if token.pos_ in ['ADJ', 'NOUN', 'PROPN'] or token.tag_ == 'VBG':
            theme_word_text = token.text
            break
            
    # Fallback to the first non-stopword if no ADJ/NOUN/VBG is found
    if theme_word_text is None:
        for token in doc:
            if not token.is_punct and not token.is_stop:
                theme_word_text = token.text
                break
                
    theme_token_id = -1
    if theme_word_text:
        # Find the start and end offsets of the theme word in the original sentence
        theme_word_match = re.search(r'\b' + re.escape(theme_word_text) + r'\b', sentence, re.IGNORECASE)
        if theme_word_match:
            start_char, end_char = theme_word_match.span()
            # Find the corresponding token index for the theme word
            for i, (offset_start, offset_end) in enumerate(tokenized_sentence.offset_mapping[0]):
                if offset_start == start_char:
                    theme_token_id = i
                    break

    # If a theme token is successfully identified, create the attention pattern.
    if theme_token_id != -1:
        # Set all tokens to attend to the theme token.
        for i in range(token_len):
            # Special tokens CLS and SEP will mostly attend to themselves and the theme token
            # to keep the pattern clean and interpretable.
            predicted_matrix[i, theme_token_id] = 1.0

    # Ensure self-attention for all tokens, which is a common pattern for this head.
    for i in range(token_len):
        predicted_matrix[i, i] = 1.0

    # Normalize the matrix rows so each row sums to 1.
    # This is standard practice for attention matrices.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows, although this is unlikely here.
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return "Sentence Theme Summarization Pattern", predicted_matrix

# Example usage (not part of the function itself, for demonstration)
# if __name__ == '__main__':
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     sentence = "The towering mountains, majestic and imposing, stood as silent sentinels, guarding the valley below, didn't they?"
#     pattern_name, predicted_matrix = get_predicted_attention_pattern(sentence, tokenizer)
#     print(f"Pattern Name: {pattern_name}")
#     print(f"Predicted Matrix Shape: {predicted_matrix.shape}")
#     # print("Predicted Attention Matrix:\n", predicted_matrix)

import numpy as np
import torch
import spacy
from transformers import BertTokenizer

def thematic_linking_pattern(sentence, tokenizer):
    """
    Generates a predicted attention matrix based on the hypothesis that the head
    is responsible for thematic and semantic linking.

    This function uses spaCy to compute the cosine similarity between word embeddings
    to approximate the attention weights. High similarity between words indicates
    a strong thematic link, which is reflected in the predicted attention matrix.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted
               attention matrix (np.ndarray of shape (token_len, token_len)).
    """
    try:
        # Load the large spaCy model for better word embeddings
        # Ensure 'en_core_web_lg' is installed: python -m spacy download en_core_web_lg
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Warning: spaCy model 'en_core_web_lg' not found. Please install it with 'python -m spacy download en_core_web_lg'.")
        # Fallback to a simpler model if the large one isn't available
        nlp = spacy.load("en_core_web_sm")

    # Tokenize the sentence with the BERT tokenizer to get the token-level structure
    bert_tokens = tokenizer.tokenize(sentence)
    full_tokens = [tokenizer.cls_token] + bert_tokens + [tokenizer.sep_token]
    token_len = len(full_tokens)

    # Use spaCy to process the sentence and get a Doc object
    doc = nlp(sentence)
    
    # Map spaCy tokens to BERT tokens. This is a key step to align the embeddings.
    # BERT tokenizer uses subwords (e.g., 'unexpected' -> 'un', '##expected').
    # We'll use the spaCy token's embedding for all its corresponding BERT subwords.
    spacy_token_map = {}
    bert_idx = 1 # Start after [CLS]
    for spacy_token in doc:
        # Get the BERT tokens for the current spaCy token
        subwords = tokenizer.tokenize(spacy_token.text)
        for _ in subwords:
            # Assign the spaCy token's embedding to each of its BERT subwords
            if bert_idx < token_len - 1:
                spacy_token_map[bert_idx] = spacy_token
            bert_idx += 1

    # Create the predicted matrix
    predicted_matrix = np.zeros((token_len, token_len), dtype=np.float32)

    # Calculate cosine similarity for all token pairs
    for i in range(1, token_len - 1):
        for j in range(1, token_len - 1):
            # Check if spaCy tokens exist for both BERT token indices
            if i in spacy_token_map and j in spacy_token_map:
                token_i = spacy_token_map[i]
                token_j = spacy_token_map[j]
                
                # Check for vector existence before calculating similarity
                if token_i.has_vector and token_j.has_vector:
                    # Calculate cosine similarity using spaCy's built-in method
                    similarity = token_i.similarity(token_j)
                    # Normalize similarity to be in [0, 1] for attention weights
                    # spaCy similarity is already in a similar range [-1, 1],
                    # but we'll scale it for our purposes.
                    predicted_matrix[i, j] = max(0, similarity) # We only care about positive similarity

    # Add self-attention for all tokens
    for i in range(token_len):
        predicted_matrix[i, i] = 1.0

    # Handle [CLS] and [SEP] tokens. A simple heuristic is to give them self-attention
    # and a small, uniform attention to all other tokens.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[-1, -1] = 1.0

    # Normalize each row (from_token) to sum to 1. This mimics the softmax
    # operation in a real attention head.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows
    row_sums[row_sums == 0] = 1
    normalized_matrix = predicted_matrix / row_sums

    return "Thematic/Semantic Linking Pattern", normalized_matrix

import numpy as np
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def list_alignment_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for a head responsible for aligning items in a list.

    The function identifies comma-separated nouns, verbs, and adjectives that are part of
    a list and assigns attention to connect them.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer used for the model (e.g., Hugging Face's tokenizer).

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    token_ids = toks.input_ids[0]
    word_ids = toks.word_ids()
    len_seq = len(token_ids)
    
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Add self-attention for special tokens
    predicted_matrix[0, 0] = 1 # CLS token
    predicted_matrix[len_seq - 1, len_seq - 1] = 1 # SEP token

    doc = nlp(sentence)
    
    # Identify and align list items
    list_items = []
    current_list = []
    
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and token.text != "and":
            current_list.append(token)
        elif token.text in [",", "and", "or", ":"] and len(current_list) > 1:
            list_items.append(current_list)
            current_list = []
    if len(current_list) > 1:
        list_items.append(current_list)
        
    for current_list in list_items:
        for i in range(len(current_list)):
            for j in range(len(current_list)):
                if i != j:
                    from_token = current_list[i]
                    to_token = current_list[j]
                    
                    # Find the token indices for the start and end of the words
                    from_token_start_id = word_ids.index(from_token.i)
                    from_token_end_id = len(word_ids) - 1 - word_ids[::-1].index(from_token.i)
                    
                    to_token_start_id = word_ids.index(to_token.i)
                    to_token_end_id = len(word_ids) - 1 - word_ids[::-1].index(to_token.i)
                    
                    # Connect all subtokens of the `from` word to all subtokens of the `to` word
                    for from_subtoken_id in range(from_token_start_id, from_token_end_id + 1):
                        for to_subtoken_id in range(to_token_start_id, to_token_end_id + 1):
                            predicted_matrix[from_subtoken_id, to_subtoken_id] += 1
    
    # Normalize the matrix to simulate uniform attention
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Enumeration/List Alignment Pattern', predicted_matrix

import numpy as np
import spacy

def thematic_linking_pattern(sentence: str, tokenizer):
    """
    Generates a rule-encoded attention matrix for the "Thematic Linking Pattern"
    found in Layer 0, Head 9.

    This pattern hypothesizes that the head is responsible for linking related
    nouns in a sentence, often lists of items, to a central, thematic noun.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A BERT tokenizer instance (e.g., BertTokenizer).

    Returns:
    - tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Load spaCy model for linguistic analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, download it and then try again
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(sentence)
    
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    token_len = len(input_ids)
    
    # Get word IDs to map spaCy tokens to tokenizer tokens
    word_ids = tokens.word_ids()

    predicted_matrix = np.zeros((token_len, token_len))

    # A dictionary to store the main noun for each thematic group
    thematic_nouns = {}

    # Identify potential central nouns for thematic groups (e.g., head of a list)
    for token in doc:
        # Check for list-like structures and main nouns
        # A simple heuristic: find nouns that are subjects or objects, or that are
        # followed by a list of nouns.
        if token.pos_ in ("NOUN", "PROPN"):
            # A simple rule to identify the "head" of a potential thematic group
            # Check if the token is followed by a colon or a comma,
            # or if it's the subject of the sentence.
            if any(child.dep_ in ("punct", "conj") for child in token.children) or token.dep_ in ("nsubj", "dobj"):
                thematic_nouns[token.text.lower()] = token.text

    # Identify tokens that are part of a list or are related to a central noun
    related_nouns = []
    # A simple heuristic to find nouns in a list: look for coordinating conjunctions
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("conj"):
            related_nouns.append(token)
    
    # Map spaCy tokens to a set of BERT token indices
    def get_token_indices(spacy_token):
        return [i for i, id in enumerate(word_ids) if id is not None and doc[id].text == spacy_token.text]

    # Rule encoding: link related nouns to their thematic head
    for related_token in related_nouns:
        # Find the head of the list
        head_token = related_token.head
        if head_token.text.lower() in thematic_nouns:
            head_indices = get_token_indices(head_token)
            related_indices = get_token_indices(related_token)
            
            for from_idx in related_indices:
                for to_idx in head_indices:
                    predicted_matrix[from_idx, to_idx] = 1.0

            # Add attention from the head to its related nouns for a bidirectional link
            for to_idx in related_indices:
                for from_idx in head_indices:
                    predicted_matrix[from_idx, to_idx] = 1.0

    # Ensure CLS and SEP tokens have self-attention
    predicted_matrix[0, 0] = 1.0  # CLS token
    if token_len > 1:
        predicted_matrix[-1, -1] = 1.0 # SEP token

    # Normalize matrix to create a uniform attention distribution per row
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1 
    normalized_matrix = predicted_matrix / row_sums

    return 'Thematic Linking Pattern', normalized_matrix

from typing import Tuple

def word_piece_alignment_pattern(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Predicts an attention matrix for the Word-Piece Alignment pattern.

    This pattern is characterized by high attention from subsequent subword tokens
    of a word to the first subword token of that word. The function encodes this
    rule into a matrix, generalizing the observed behavior.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get word IDs
    tokenized_input = tokenizer(
        [sentence],
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    input_ids = tokenized_input.input_ids[0]
    seq_len = len(input_ids)
    
    # Initialize the predicted matrix with zeros
    predicted_matrix = np.zeros((seq_len, seq_len))
    
    # Get word IDs to identify word boundaries. This can be None if the input is empty.
    # We must handle this edge case to prevent the error.
    word_ids_list = tokenized_input.word_ids()
    if not word_ids_list or word_ids_list[0] is None:
        # If no word IDs are returned, just return a zero matrix (or self-attention
        # for special tokens if they exist)
        for i in range(seq_len):
            predicted_matrix[i, i] = 1.0
        return 'Word-Piece Alignment Pattern', predicted_matrix

    word_ids = word_ids_list[0]

    # Map each word_id to the index of its first token
    first_token_map = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in first_token_map:
            first_token_map[word_id] = i

    # Populate the matrix based on the first token of each word
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            # Special tokens attend to themselves
            predicted_matrix[i, i] = 1.0
        else:
            # All tokens of a word attend to the first token of that word
            first_token_idx = first_token_map[word_id]
            predicted_matrix[i, first_token_idx] = 1.0

    # Normalize each row to ensure the attention weights sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero by replacing zeros in row_sums with ones
    row_sums[row_sums == 0] = 1
    normalized_matrix = predicted_matrix / row_sums
    
    return 'Word-Piece Alignment Pattern', normalized_matrix

def subword_token_reassembly(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 0, Head 11 of BERT, which is
    believed to be responsible for reassembling sub-word tokens.

    The function generates a predicted attention matrix where a token with the '##'
    prefix pays attention to the token immediately preceding it. This simulates the
    pattern observed in the data for re-linking word fragments.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for
                                                     tokenizing the sentence.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get input IDs and word IDs
    encoded = tokenizer(sentence, return_tensors='pt')
    input_ids = encoded['input_ids'][0]
    word_ids = encoded.word_ids()
    token_len = len(input_ids)
    
    # Initialize a zero matrix for attention
    predicted_matrix = np.zeros((token_len, token_len))

    # Identify sub-word tokens and their preceding tokens
    for i in range(1, token_len):
        token_str = tokenizer.decode(input_ids[i])
        
        # Check for sub-word token marker
        if token_str.startswith('##'):
            # The token will attend to its immediate predecessor.
            # In BERT's tokenization, the preceding token is part of the same word.
            predicted_matrix[i, i-1] = 1.0

    # Ensure CLS and SEP tokens have self-attention, a common base pattern.
    predicted_matrix[0, 0] = 1.0  # [CLS] token
    predicted_matrix[-1, -1] = 1.0  # [SEP] token
    
    # Normalize the attention matrix by row (results in uniform attention for each row)
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows with no attention.
    predicted_matrix = np.divide(predicted_matrix, row_sums, where=row_sums != 0)

    return 'Sub-word Token Reassembly Pattern', predicted_matrix

# Example of how to use the function:
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_sentence = "The unexpected downpour ruined our day."
    
    pattern_name, predicted_attention = subword_token_reassembly(test_sentence, tokenizer)
    
    print(f"Pattern Name: {pattern_name}")
    print("Predicted Attention Matrix:")
    print(predicted_attention)
    
    # To show the token-level pattern clearly
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(test_sentence)['input_ids'])
    print("\nTokens:", tokens)
    print("Attention from '##pour' to 'down':", predicted_attention[5, 4])
    print("Attention from '##ed' to 'ruin':", predicted_attention[7, 6])