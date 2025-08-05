import numpy as np

def generate_next_token_attention_matrix(sentence, tokenizer):
  """
  Generates a predicted attention matrix for Layer 2, Head 9, based on the hypothesis
  that the head's responsibility is to link each token to the next token in the sequence.

  Args:
    sentence (str): The input sentence.
    tokenizer: The tokenizer object (e.g., from the Hugging Face library).

  Returns:
    tuple: A tuple containing the name of the pattern and the predicted attention matrix.
  """

  # Tokenize the sentence and get the number of tokens
  toks = tokenizer([sentence], return_tensors="pt")
  len_seq = len(toks.input_ids[0])
  
  # Initialize an attention matrix of zeros
  predicted_matrix = np.zeros((len_seq, len_seq))

  # The core pattern: attend from each token to the next
  # Loop from the first token up to the second-to-last token
  for i in range(len_seq - 1):
    predicted_matrix[i, i + 1] = 1

  # Based on the data, the final token also has attention, often to the EOS or
  # a preceding token. For simplicity and to match the pattern's generalization,
  # we'll model this as self-attention for the final token.
  # The first token also attends to itself (CLS token)
  predicted_matrix[0, 0] = 1
  predicted_matrix[len_seq - 1, len_seq - 1] = 1
  
  # Return the pattern name and the unnormalized matrix.
  # Normalization is often an implicit part of the attention mechanism,
  # but returning the raw rule-encoded matrix is more transparent.
  return "Next Token Attention Pattern", predicted_matrix