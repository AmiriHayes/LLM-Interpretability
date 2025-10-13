import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spaCy model for Named Entity Recognition
en_nlp = spacy.load('en_core_web_sm')

def ner_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to analyze the sentence and extract named entities
    doc = en_nlp(sentence)
    named_entities = [(ent.start, ent.end) for ent in doc.ents]

    # Construct attention pattern based on named entities
    for start, end in named_entities:
        start_tok_id = toks.word_to_tokens(toks.tokenize(sentence[start:end], is_split_into_words=False)).tokens()[0]
        end_tok_id = toks.word_to_tokens(toks.tokenize(sentence[end], is_split_into_words=False)).tokens()[-1]
        for i in range(start_tok_id, end_tok_id + 1):
            for j in range(start_tok_id, end_tok_id + 1):
                out[i, j] = 1.0

    # Normalize each row and ensure no row is all zero by defaulting some attention to the last token (typically punctuation for end of sentence)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Named Entity Recognition Based Attention", out