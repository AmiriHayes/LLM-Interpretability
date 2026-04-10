from transformers import PreTrainedTokenizerBase
import numpy as np
def semantic_role_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    for i, word in enumerate(words):
        if word.endswith('ly') or word.endswith('ing'):
            for j, context_word in enumerate(words):
                if context_word in ['of', 'with', 'for', 'to', 'and', 'in']:
                    verb_index = sentence.index(word) - sentence[:sentence.index(word)].count(' ') # Obtain token index for word
                    context_index = sentence.index(context_word) - sentence[:sentence.index(context_word)].count(' ') # Context word index
                    if verb_index > 0 and context_index > 0 and verb_index < len_seq and context_index < len_seq:
                        out[verb_index, context_index] = 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Semantic Role Association", out