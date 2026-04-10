import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the English NLP model from spaCy
en_nlp = spacy.load('en_core_web_sm')


def pronoun_antecedent_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = en_nlp(sentence)

    token_to_id = {tok.text: tok.i+1 for tok in doc}  # Map words to index + 1 (CLS is at index 0)

    # Define two lists for pronouns and their likely antecedents
    pronouns = {'she', 'her', 'it', 'he', 'they', 'them', 'we', 'you', 'i'}
    skip_tokens = {'and', 'or', 'but', ',', '.', '!', '?', 'with', 'on', 'in', 'at', 'for', 'to'}

    # Track potential antecedents
    antecedents = []

    for token in doc:
        tok_text = token.text.lower()
        if tok_text in pronouns and antecedents:
            # Focus on strong connections to closest antecedent
            out[token_to_id[token.text] - 1, token_to_id[antecedents[-1].text] - 1] = 1
        elif tok_text not in skip_tokens:
            # Record potential antecedents, avoiding punctuation
            antecedents.append(token)

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun-Antecedent Resolution", out