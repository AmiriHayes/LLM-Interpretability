import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')  # load English tokenizer, tagger, parser, NER, and word vectors


def conjunction_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Generate a spaCy doc for dependency parsing
    doc = nlp(' '.join(words))

    # Align token positions between spaCy and the tokenizer
    token_mapping = {token.i: i for i, token in enumerate(doc)}

    for token in doc:
        # Identify conjunctions or subordinating conjunctions
        if token.dep_ in ['cc', 'mark']:  # 'cc' is coordinating conjunction, 'mark' is for 'because', 'although', etc.
            parent_index = token_mapping.get(token.head.i, -1)
            out[token.i, parent_index] = 1
            out[parent_index, token.i] = 1

            # Set additional structure relevant to sentence as a whole or specific conjunction relations
            if token.dep_ == 'cc':
                # Connect the token to its nearest linked verb or noun (head) in the dependency tree
                for child in token.children:
                    child_index = token_mapping.get(child.i, -1)
                    out[child_index, token.i] = 1
                    out[token.i, child_index] = 1

    # Ensure at least one attention per row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize rows
    out /= out.sum(axis=1, keepdims=True)

    return "Conjunction and Subordinating Conjunction Dependency Linking", out