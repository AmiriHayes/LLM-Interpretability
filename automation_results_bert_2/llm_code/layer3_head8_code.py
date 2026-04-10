import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")


def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy to get word tokens and coreferences
    doc = nlp(sentence)

    word_to_token = {}
    idx = 1  # Start from 1 as 0 is [CLS]
    for token in doc:
        span = token.idx, token.idx + len(token.text)
        sub_tokens = toks.word_to_tokens(0, start=span[0], end=span[1])
        if sub_tokens:
            word_to_token[token.text.lower()] = (sub_tokens.start, sub_tokens.end)

    for cluster in doc._.coref_clusters:
        # Get representative mentions for coreferent mentions
        main_mention = cluster.main.text.lower().strip()
        for mention in cluster.mentions:
            mention_text = mention.text.lower().strip()
            if main_mention in word_to_token and mention_text in word_to_token:
                start_main, end_main = word_to_token[main_mention]
                start_mention, end_mention = word_to_token[mention_text]
                # Create attention links
                for i in range(start_main, end_main):
                    for j in range(start_mention, end_mention):
                        out[i, j] = 1
                        out[j, i] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Set to attend to [SEP]

    return "Coreference Resolution Pattern", out

