import numpy as np
from transformers import PreTrainedTokenizerBase


def math_expression_highlight(sentence: str, tokenizer: PreTrainedTokenizerBase):
    """
    Model attention pattern for highlighting mathematical expressions. This function creates an attention map with higher
    focus on tokens that part of mathematical expressions and assumes some form of parentheses matching and other
    identifiable math symbols.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_ids = toks.data['input_ids'].tolist()[0]
    # Create a dictionary mapping token indices to token strings
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Define mathematical symbols and number-related tokens
    math_related_tokens = {'$': 1, '\\': 2, 'sum': 3, '\theta': 4, '^': 5, '(', ')', '*', '+', '-', '\le', '\ge', 'cdot'}
    number_tokens = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '##1', '##2', '##3', '##4', '##5', '##6', '##7', '##8', '##9'}

    # Initialize stacks for keeping track of parentheses
    parentheses_stack = []

    # Highlight attention for math-related parts
    for i, token in enumerate(tokens):
        # Uniformly increase attention to known math symbols and numbers
        if token in math_related_tokens or token in number_tokens:
            out[i, i] = 0.5

        # Increase for matched parentheses and braces
        if token == '(' or token == '{':
            parentheses_stack.append(i)
        elif (token == ')' or token == '}') and parentheses_stack:
            matched_indx = parentheses_stack.pop()
            out[matched_indx, i] = 0.9
            out[i, matched_indx] = 0.9

    # Set attention for start and end tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Regularize row sum
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1)

    return "Mathematical Expression Highlight", out
