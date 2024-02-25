#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from pathlib import Path
from typing import Union
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        print('-----------------Loading adapter------------------')
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer



def load_model(model_dir: str):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    return model, tokenizer
def generate_response(model, tokenizer, prompt) -> str:
    response, _ = model.chat(tokenizer, prompt)
    return response

# 调用函数的示例
if __name__ == '__main__':

    model_directory = "/home/gang/Documents/climate/chatglm3-6b-climate-output/checkpoint-3000"  # 这里替换成你的模型目录路径
    model, tokenizer = load_model(model_directory)
    user_prompt = "你是谁"  # 这里替换成你想要输入给模型的提示文本
    print(generate_response(model, tokenizer, user_prompt))
    user_prompt = "你能为我做什么？"  # 这里替换成你想要输入给模型的提示文本
    print(generate_response(model, tokenizer, user_prompt))

