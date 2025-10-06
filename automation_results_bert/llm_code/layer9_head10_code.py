from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')


def named_entities_and_relations(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Focuses on named entities and their relations like people and objects.

    Args:
        sentence (str): The input sentence.
        tokenizer (PreTrainedTokenizerBase): A tokenizer object to tokenize the input sentence.

    Returns:
        Tuple[str, np.ndarray]: A name indicating the pattern detected and a prediction matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to identify named entities
    doc = nlp(sentence)
    ents = {ent.text: ent for ent in doc.ents}

    # Map token indices from spaCy to BERT tokenizer
    word_ids = toks.word_ids(batch_index=0)
    spaCy_to_bert_idx = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            word = toks.tokens(0)[i]
            if word.startswith("##"):
                continue
            spaCy_to_bert_idx[word_id] = i

    # Create attention pattern based on named entities
    for ent_label, ent in ents.items():
        ent_indices = [spaCy_to_bert_idx.get(n, None) for n in range(ent.start, ent.end)]
        # Remove None values
        ent_indices = [i for i in ent_indices if i is not None]

        for idx1 in ent_indices:
            for idx2 in ent_indices:
                out[idx1, idx2] = 1.0
            # Also, add attention between entity and tokens with some attention weights in data
            for token_idx in range(1, len_seq - 1):
                if ent.root.head.i == token_idx:
                    out[idx1, token_idx] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize row to sum to 1 avoiding division by zero
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Focus on Named Entities and Their Relations', out