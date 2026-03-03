from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def get_model_and_tokenizer(model_name):
    if model_name == "bert":
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    
    elif model_name == "gpt-2":
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model

    elif model_name == "tiny-llama":
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True,
        )
        model.eval()
        return tokenizer, model