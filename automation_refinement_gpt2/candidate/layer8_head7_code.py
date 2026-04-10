from transformers import PreTrainedTokenizerBase
import numpy as np

def topic_maintenance_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simplified rule: Assume the first token or first substantive token is the topic;
    # Subsequent tokens often refer back to this token in varying degrees.

    # Usually, the first token is a determiner (The, A, etc.).
    # We identify non-trivial topics (nouns, pronouns) as those that control attention 
    # throughout a sentence.

    # Get tokens and determine topic
    tokens = [tok.lower() for tok in sentence.split()]
    potential_topics = {0}  # Start with the first word being the topic

    # Consider first noun as more definitive topic
    for i, tok in enumerate(toks.input_ids[0]):
        if i > 0 and tokens[i][0].isalpha() and tokens[i-1][-1] != ',':
            potential_topics.add(i)

    for column in range(len_seq):
        for topic_index in potential_topics:
            out[topic_index, column] = 1.0 / len(potential_topics)

    # Ensure normalization so every row sums to 1 (avoid trivial attention with numerical issues)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] /= out[row].sum()

    return "Topic Maintenance Attention", out