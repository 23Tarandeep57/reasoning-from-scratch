from reasoning_from_scratch.qwen3 import download_qwen3_small
import torch
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, Qwen3Tokenizer
from pathlib import Path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

download_qwen3_small(kind='base', tokenizer_only=False, out_dir = 'qwen3')
model_path = Path("qwen3") / 'qwen3-0.6B-base.pth'
tokenizer_path = Path("qwen3") / "tokenizer-base.json"

tokenizer = Qwen3Tokenizer(tokenizer_file_path = tokenizer_path)

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))

def generate_text_basic(
        model,
        token_ids,
        max_new_tokens,
        eos_token_id = tokenizer.eos_token_id
        ):
    input_len = token_ids.shape[1]
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        out_token = torch.argmax(out , dim = -1, keepdims = True)

        if eos_token_id is not None and out_token == eos_token_id:
            break;

        token_ids = torch.concat([token_ids, out_token], dim = 1)
    return token_ids[:, input_len:]

prompt = "write a short poem on defiance and breaking out of conformity."

max_tokens = 100
input_token_ids = torch.tensor(tokenizer.encode(prompt), device = device).unsqueeze(0)

out_ids = generate_text_basic(model, input_token_ids, max_new_tokens = max_tokens).squeeze(0)
out_text = tokenizer.decode(out_ids.tolist())
print(out_text)
        
