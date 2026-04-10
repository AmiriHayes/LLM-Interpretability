import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spaCy model (English)
nlp = spacy.load('en_core_web_sm')

def context_semantic_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy to get semantic relationships
    doc = nlp(sentence)

    # Map spaCy tokens to indices in the tokenized output
    word_to_index = {}
    spacy_index = 0
    for tok_index in range(len_seq):
        while spacy_index < len(doc):
            if toks.input_ids[0][tok_index].item() == tokenizer.convert_tokens_to_ids([doc[spacy_index].text])[0]:
                word_to_index[spacy_index] = tok_index
                spacy_index += 1
                break
            spacy_index += 1

    # Assign attention between nouns and their associated verbs
    for token in doc:
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == 'VERB' and child.dep_ in {'nsubj', 'dobj'}:
                    if token.i in word_to_index and child.i in word_to_index:
                        out[word_to_index[token.i], word_to_index[child.i]] = 1
                        out[word_to_index[child.i], word_to_index[token.i]] = 1

    # Assign CLS and SEP token self-attention
    out[0, 0] = 1  # CLS token
    out[-1, 0] = 1  # SEP token

    # Normalize the attention matrix row-wise
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Contextual Semantic Linkage with Noun-Verb Associations", out