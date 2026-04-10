import numpy as np
from transformers import PreTrainedTokenizerBase

def arithmetic_sequence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    def find_arithmetic_numbers(tokens):
        """
        Identify pairs of numbers that form an arithmetic sequence in the sentence.
        """
        indices = []
        numbers = []
        for i, token in enumerate(tokens):
            if token.isdigit():
                numbers.append((i, int(token)))
        if len(numbers) < 3:
            return indices

        for i in range(len(numbers) - 2):
            a, b, c = numbers[i][1], numbers[i + 1][1], numbers[i + 2][1]
            if b - a == c - b:
                indices.append((numbers[i][0], numbers[i + 1][0], numbers[i + 2][0]))
        return indices

    tokens = sentence.split()
    arithmetic_triples = find_arithmetic_numbers(tokens)

    for start, middle, end in arithmetic_triples:
        out[start, middle] = 1
        out[middle, end] = 1

    return "Arithmetic Sequence Pattern", out