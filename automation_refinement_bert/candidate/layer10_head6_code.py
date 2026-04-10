from transformers import PreTrainedTokenizerBase
import numpy as np

def arithmetic_expression_parsing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence to identify potential numbers and operators
    operators = {'+', '-', '*', '/', '=', '(', ')', ',', '[', ']', '{', '}', '<', '<=', '>', '>='}
    token_types = [(i, token, (token.isnumeric() or token in operators)) for i, token in enumerate(toks.tokens())]

    # Identify numbers and operators and assign attention based on type
    for i, token, is_num_or_op in token_types:
        if is_num_or_op:
            for j, _, is_num_or_op_j in token_types:
                if is_num_or_op_j:
                    out[i, j] = 1

    # Ensure the beginning [CLS] and ending [SEP] token attend to the start and end
    out[0, 0] = out[-1, 0] = 1

    # Normalize the attention matrix by rows
    out = out / out.sum(axis=1, keepdims=True)

    return "Arithmetic Expression Parsing Pattern", out