import numpy as np
from transformers import PreTrainedTokenizerBase


def semantic_association_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Dictionary to maintain a mapping of connectable tokens based on observation 
    # Observed associations (semantic roles, synonymy, thematic relations, etc.)
    semantic_mappings = {
        'painting': ['hue', 'vibrant', 'orange', 'pink', 'purple'],
        'intricate': ['details', 'problem', 'tapestry', 'design'],
        'joy': ['played', 'splash', 'puddle'],
        'secrets': ['whisper', 'times', 'past'],
        'transport': ['world', 'adventure', 'mystery'],
        'anticipation': ['hopeful', 'journey'],
        'aroma': ['bread', 'kitchen', 'hungry'],
        'side': ['other'],
        'tapestry': ['details', 'skill', 'precision'],
        'succeed': ['give', 'dreams'],
        'symphony': ['sounds', 'sights', 'smells'],
        'tell': ['could', 'please', 'way'],
        'stood': ['whisper', 'secrets', 'sentinel'],
        'language': ['learning', 'challenging'],
        'promise': ['day', 'remembered'],
        'evidence': ['footprints', 'smell', 'scent'],
        'performance': ['first', 'solo', 'prepared', 'very']
    }

    # Build predicted attention according to observed patterns
    for word_idx, word in enumerate(sentence.lower().split()):
        for key, associates in semantic_mappings.items():
            if key in word or any(associate in word for associate in associates):
                key_index = next((i for i, t in enumerate(sentence.lower().split()) if key in t), None)
                if key_index:
                    out[word_idx, key_index] = 1 

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Semantic Association Pattern", out