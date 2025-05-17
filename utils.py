from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

api_key = ""


def get_model_path(model_name):
    if model_name == "qwen2.5-7b-instruct":
        return ""
    elif model_name == "meta-llama-3-8b-instruct":
        return ""
    else:
        raise ValueError(f"unknown model {model_name}.")


def get_model(model_name, max_new_tokens, temperature):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if isinstance(max_new_tokens, str):
        max_new_tokens = int(max_new_tokens) 
    generation_config = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 1e-5),
        "top_p": 0.9,
        "top_k": 50,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return model, tokenizer, generation_config


def get_prompt(tokenizer, role, exp):
    messages = []
    messages.append({"role": "system", "content": role})
    messages.append({"role": "user", "content": exp})

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        text = "\n".join([msg["content"] for msg in messages])
        inputs = tokenizer(text, return_tensors="pt").input_ids.squeeze().tolist()

    return inputs


def predict(model, tokenizer, generation_config, role, exp):
    model.eval()
    with torch.no_grad():
        input_ids = get_prompt(tokenizer, role, exp)
        input_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        output = model.generate(input_ids, **generation_config)
        output = output[0][input_len:]
    return tokenizer.decode(output, skip_special_tokens=True)
