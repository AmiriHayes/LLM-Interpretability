import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def needle_action_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and align the tokens with their word pieces
    token_words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    doc = nlp(sentence)

    # Create a mapping between token positions in the sentence and word positions in spaCy
    token_word_map = {}
    tok_idx = 1  # Start after [CLS]
    for i, token_obj in enumerate(doc):
        # Accommodate word tokens potentially split into sub-tokens in BERT
        full_word = ''
        while tok_idx < len_seq - 1 and not full_word.startswith(token_obj.text):
            token_part = token_words[tok_idx].replace("##", "")
            full_word += token_part
            token_word_map[tok_idx] = i
            tok_idx += 1

    # Determine specific categories
    needle_pos = [i for i, token in enumerate(doc) if token.text.lower() == 'needle']
    action_words = {'share', 'fix', 'found', 'help', 'sew', 'work'}
    action_pos = [i for i, token in enumerate(doc) if token.lemma_ in action_words]

    # Fill the matrix based on identified roles
    for tok_idx, word_idx in token_word_map.items():
        if word_idx in needle_pos:
            for target_tok_idx in token_word_map:
                if token_word_map[target_tok_idx] in action_pos:
                    out[tok_idx, target_tok_idx] = 1
        elif word_idx in action_pos:
            for target_tok_idx in token_word_map:
                if token_word_map[target_tok_idx] in needle_pos:
                    out[tok_idx, target_tok_idx] = 1

    # Ensure non-zero attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Needle and Action-Related Attention", out