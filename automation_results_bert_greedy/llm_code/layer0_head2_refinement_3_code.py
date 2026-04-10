from transformers import PreTrainedTokenizerBase
import numpy as np

def article_adjective_noun_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:  # Removing Tuple typology for clarity
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Check for words by creating alignment of tokenizer and nltk pos tagging
    words = sentence.split()

    # Corresponds sentence tokens to nltk words
    bpe_to_words_alignment = []
    words_index = 0
    for tok in toks.tokens()[0]:
        if tok.startswith("##"):
            bpe_to_words_alignment.append(words_index - 1)
        elif tok in ("[CLS]", "[SEP]"):
            bpe_to_words_alignment.append(None)  # Separator tokens
        else:
            bpe_to_words_alignment.append(words_index)
            words_index += 1

    article_tokens = ("a", "the")
    # Traverse through and find tokens
    for i in range(len_seq):
        if bpe_to_words_alignment[i] is not None:
            current_word = words[bpe_to_words_alignment[i]]

            if current_word in article_tokens:
                # Attend to the next word, often a noun or adjective
                j = i + 1
                while j < len_seq and bpe_to_words_alignment[j] is not None:
                    next_word = words[bpe_to_words_alignment[j]]
                    # Skip conjunctions
                    if next_word not in ("and", ","):
                        out[i, j] = 1
                        break
                    j += 1

            elif current_word.lower() in ("a", "the"):
                # Deal with lowercase articles for robustness
                j = i + 1
                while j < len_seq and bpe_to_words_alignment[j] is not None:
                    next_word = words[bpe_to_words_alignment[j]]
                    if next_word.lower() in article_tokens:
                        out[i, j] = 1
                        break
                    j += 1

            elif any(adj == current_word.lower() for adj in ["little", "big", "happy", "healthy"]):
                # Attend to the subsequent noun
                j = i + 1
                while j < len_seq and bpe_to_words_alignment[j] is not None:
                    out[i, j] = 1
                    break
                j += 1

    # Default attention for CLS and SEP tokens
    out[0, 0] = 1.0
    out[-1, -1] = 1.0
    out[:, -1] = 1.0  # Ensure no row is all zeros
    out = out / out.sum(axis=1, keepdims=True) + 1e-4

    return "Article-Noun and Adjective-Noun Linking", out