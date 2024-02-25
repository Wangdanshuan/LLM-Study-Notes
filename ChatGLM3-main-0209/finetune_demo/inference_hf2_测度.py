#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from pathlib import Path
from typing import Union
import random
import pandas as pd
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
import tqdm
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
    response, history = model.chat(tokenizer, prompt)
    return response, history

# 调用函数的示例
import json
if __name__ == '__main__':

    model_directory = "/home/gang/Documents/climate/chatglm3-6b-climate-output/checkpoint-2000"  # 这里替换成你的模型目录路径
    model, tokenizer = load_model(model_directory)
    mydf = pd.read_excel('所有MDA章节级别的无标签数据集.xlsx')
    mydf.dropna(inplace=True, axis=0)

    # 打开文件以追加模式，确保每次运行代码时不会覆盖原有数据
    with open('MDA_measure_records_trans.jsonl', 'a') as file:
        for index, row in tqdm(mydf.iterrows(), total=mydf.shape[0]):
            text = row['MDA']
            年份 = row['年份']
            月份 = row['月份']
            企业 = row['企业']
            insturction = "请帮我分析以下文本，找出提及气候转型风险的段落，并按此格式回答：‘[段落编号]/无’。 文本如下："
            user_prompt = insturction + "<" + text + ">"
            # 假设generate_response是一个函数，用于生成响应。这里需要确保你有这个函数的实现。
            response, history = generate_response(model, tokenizer, user_prompt)
            print(history)
            # 构建一个字典来保存当前行的数据和响应
            data = {
                'MDA': text,
                'response': response,
                '年份': 年份,
                '月份': 月份,
                '企业': 企业,
            }
            # 将字典转换为JSON字符串并追加到文件
            file.write(json.dumps(data) + '\n')



    with open('MDA_measure_records_physical.jsonl', 'a') as file:
        for index, row in tqdm(mydf.iterrows(), total=mydf.shape[0]):
            text = row['MDA']
            年份 = row['年份']
            月份 = row['月份']
            企业 = row['企业']
            insturction = "请帮我分析以下文本，找出提及气候物理风险的段落，并按此格式回答：‘[段落编号]/无’。 文本如下："
            user_prompt = insturction + "<" + text + ">"
            response, history = generate_response(model, tokenizer, user_prompt)
            print(history)
            # 构建一个字典来保存当前行的数据和响应
            data = {
                'MDA': text,
                'response': response,
                '年份': 年份,
                '月份': 月份,
                '企业': 企业,
            }
            # 将字典转换为JSON字符串并追加到文件
            file.write(json.dumps(data) + '\n')