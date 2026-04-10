import numpy as np
from transformers import PreTrainedTokenizerBase


def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Create a mapping from word positions to token indexes
    word_to_token_map = {}
    for i, tok_id in enumerate(toks.input_ids[0]):
        token_word = tokenizer.decode(tok_id)
        if token_word.startswith("##"):
            token_word = token_word[2:]  # Handle subword tokens
        word_to_token_map[token_word] = i

    # Hypothesize coreference by connecting pronouns to their likely referents
    pronoun_positions = []
    previous_mentions = {}

    for i, word in enumerate(words):
        # Identify pronouns and connect them to prior mentions
        if word.lower() in {"he", "she", "it", "they", "her", "his", "their"}:
            pronoun_positions.append(i)
            # Heuristic: Use last mentioned entity from previous tokens
            if previous_mentions:
                last_mention_index = max(previous_mentions.values())
                if last_mention_index < len_seq:
                    out[word_to_token_map[word], last_mention_index] = 1
        else:
            # For non-pronoun words, add them to previous mentions
            previous_mentions[word.lower()] = word_to_token_map[word]

    # Adding self-attention to ensure no all-zero rows
    for i in range(len_seq):
        if out[i].sum() == 0:
            out[i, i] = 1.0

    # Normalize the rows to sum to 1 for uniform attention
    out += 1e-4  # Avoid division by zero errors
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreference Resolution Pattern", out