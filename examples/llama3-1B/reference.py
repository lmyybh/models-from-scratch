# 关闭日志
import logging
from modelscope.utils.logger import get_logger
logger = get_logger(log_level=logging.ERROR)

import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM

device = torch.device(0)

# 导入 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("LLM-Research/Llama-3.2-1B").to(device)

# 输入的 prompt
prompt = "The highest mountain in the world is"
data = tokenizer.batch_encode_plus([prompt], return_tensors="pt")

# 文本生成
outputs = model.generate(
    data["input_ids"].to(device),
    attention_mask=data["attention_mask"].to(device),
    pad_token_id=tokenizer.pad_token_type_id,
    top_k=1, # 规定选择最高概率的 token
    max_length=128, # 最大文本长度
)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)
