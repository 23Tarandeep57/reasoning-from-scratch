from reasoning_from_scratch.qwen3 import download_qwen3_small
import torch
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, Qwen3Tokenizer
from pathlib import Path
import sys
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

download_qwen3_small(kind='base', tokenizer_only=False, out_dir = 'qwen3')
model_path = Path("qwen3") / 'qwen3-0.6B-base.pth'
tokenizer_path = Path("qwen3") / "tokenizer-base.json"

tokenizer = Qwen3Tokenizer(tokenizer_file_path = tokenizer_path)

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))

def generate_text_basic_streaming(
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

        if eos_token_id is not None and out_token.item() == eos_token_id:
            break;

        token_ids = torch.concat([token_ids, out_token], dim = 1)
        yield out_token

prompt = "explain LLMs in single sentence."
max_new_tokens = 100
input_token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
# output_ids = generate_text_basic(model, input_token_ids, max_new_tokens)

import time
start_time = time.time()
num_tokens = 0;
for token in generate_text_basic_streaming(model, input_token_ids, max_new_tokens):
    # print(token)
    output_ids = token.squeeze(0)
    out_text = tokenizer.decode(output_ids.tolist())
    print(out_text, end = "", flush = True)
    num_tokens+=1
end_time = time.time() 


print(f"total time taken: {end_time - start_time:.2f}")
