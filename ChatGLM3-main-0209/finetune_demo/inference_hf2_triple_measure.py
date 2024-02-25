#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from pathlib import Path
from typing import Union
import random
import re
import pandas as pd
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from tqdm import tqdm
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

def get_segments(row):
    segment_texts = re.split(r'(\[S\d+\])', row['MDA'])
    # 构造一个字典，映射每个段落编号到其对应的文本内容
    segment_to_text = {segment_texts[i]: segment_texts[i + 1] for i in range(1, len(segment_texts), 2)}

    # 提取并排序所有段落编号，确保顺序性
    segments = list(segment_to_text.keys())
    segments.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    texts = []
    for segment in segments:
        segment_text = segment_to_text.get(segment, "")
        texts.append(segment_text)
    return texts, segments
# 调用函数的示例
import json
if __name__ == '__main__':

    instructions = [
        "气候风险可以分为两大类：气候物理风险主要指气候变化导致生产性资产直接受损的风险；气候转型风险是指在全球应对气候变化的背景下，向低碳、绿色经济过渡期间产生的财务风险和经济挑战，这类风险主要来源于政策调整、技术创新、市场变化以及社会行为和期望的变化。"
        "请问下文内容涉及了哪些气候风险类型？请从{物理风险, 转型风险, 物理风险和转型风险共存, 与气候风险无关}中选择答案",
        "随着全球气候变化的加剧，气候风险对经济和社会的影响日益显著。其中，气候物理风险涉及自然灾害如洪水、干旱和暴风雨对生产和生活资产的直接损害。另一方面，气候转型风险指的是在向低碳经济转型过程中，由于政策、技术、市场和社会预期的变化而产生的财务和经济风险。这包括但不限于碳定价、新能源技术的发展和消费者偏好的变化。"
        "基于上述描述，请判断文中提到了哪些类型的气候风险？请选择最合适的答案：{物理风险, 转型风险, 物理风险和转型风险共存, 与气候风险无关}",
        "气候变化对全球经济和社会安全构成了前所未有的挑战。具体来说，气候物理风险涵盖了由于全球温度上升导致的极端天气事件，如海平面上升、极端温度变化导致的直接物理损害。而气候转型风险则涉及在实现可持续发展目标和低碳转型过程中遇到的经济和财务挑战，这可能包括新的政府政策、绿色技术的推广以及消费模式的改变。"
        "根据以上信息，请识别文中讨论的气候风险类型。选项包括：{物理风险, 转型风险, 物理风险和转型风险共存, 与气候风险无关}",
        "在应对气候变化的过程中，区分不同类型的气候风险至关重要。气候物理风险通常指因气候变化加剧的自然灾害（如洪水、旱灾和风暴）给经济带来的直接影响。而气候转型风险则关注在向绿色经济过渡的过程中遇到的挑战，这可能涉及政策变化、技术进步、市场需求的转变以及公众意识的提高。"
        "考虑到这些因素，请指出文中讨论了以下哪种气候风险？请从以下选项中选择：{物理风险, 转型风险, 物理风险和转型风险共存, 与气候风险无关}"

    ]
    model_directory = "/home/bingxing2/home/scx6mdq/myllm/ChatGLM3-main-0209/finetune_demo/output-Triple-classification-addsample-2batch/checkpoint-15500"  # 这里替换成你的模型目录路径
    model, tokenizer = load_model(model_directory)
    mydf = pd.read_excel('/home/bingxing2/home/scx6mdq/DMA_unlabel_datasets_2048.xlsx')
    mydf.dropna(inplace=True, axis=0)
    max_records = 500
    mydf = mydf.sample(max_records, random_state=45)
    # 打开文件以追加模式，确保每次运行代码时不会覆盖原有数据
    with open('MDA_measure_triple_15500.jsonl', 'a') as file:
        for index, row in tqdm(mydf.iterrows(), total=mydf.shape[0]):
            texts, nums = get_segments(row)
            年份 = row['年份']
            月份 = row['月份']
            企业 = row['企业']
            insturction = random.choice(instructions)
            for j, text in enumerate(texts):
                user_prompt = insturction + '。需要分析的文本如下：' + '<' + text +'>'
                # 假设generate_response是一个函数，用于生成响应。这里需要确保你有这个函数的实现。
                response, history = generate_response(model, tokenizer, user_prompt)
                print(history)
                # 构建一个字典来保存当前行的数据和响应
                data = {
                    'MDA': text,
                    'response': response,
                    '文章编号': index,
                    '段落编号': j,
                    '年份': 年份,
                    '月份': 月份,
                    '企业': 企业,
                }
                # 将字典转换为JSON字符串并追加到文件
                file.write(json.dumps(data) + '\n')



