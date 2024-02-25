# 1.改写Seq2SeqTrainer

我认为这里的改写目的就是让输出的generated_tokens只有模型生成部分的tocken，对于用户输入已经过滤掉了。
以及在验证或者预测时，在输入模型之前，从inputs拿掉"output_ids"，总之看起来并不是很关键。这里inputs不包含"labels"，只有"input_ids"，然后输入模型，prediction_loss_only的设置为False因此也不需要计算损失。

```
class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids

        return loss, generated_tokens, labels
```
从模型训练保存的配置文件中我注意到，prediction_loss_only的设置为False。它是默认设置的，也就是不计算损失，inputs中不需要包括"labels"。
transformers其实很非常多的参数，一般只对其一部分设置，其余的均保持默认。
具体的参数设置可以通过如下方式查看：
        ```
         from transformers import Seq2SeqTrainingArguments
         # 查看 Seq2SeqTrainingArguments 的文档字符串
         help(Seq2SeqTrainingArguments)
        ```
        

# 2. 模型加载以及用peft封装，以进行lora微调。参考“大模型的LORA微调与加载.md”
# 3. 数据的预处理


```
def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}
```

当 `process_batch` 函数传入 `map` 函数作为其处理函数时，`process_batch` 会对数据集中的每批数据进行处理。具体来说，这个组合会对数据集进行以下操作：

1. **数据提取和转换**：`process_batch` 函数首先从每个批次中提取 `tools` 和 `conversations` 信息。然后，它根据对话内容和角色信息构建 `input_ids` 和 `loss_masks`。这包括使用分词器 (`tokenizer`) 将文本内容转换为模型可以理解的 token ID，并根据对话中的角色信息设置损失掩码（`loss_mask_val`）。

2. **特殊标记添加**：为每个对话添加特殊的开始 (`[gMASK]`) 和分隔 (`sop`) 标记，以及在对话结束时添加结束标记 (`eos_token_id`)。

3. **损失掩码处理**：通过 `loss_masks` 确定哪些 token 应计入损失计算。对于不需要计入损失的 token，使用 `-100` 作为其标签值，这在 PyTorch 中通常用于忽略特定 token 的损失。

4. **序列长度调整**：确保 `input_ids` 和 `labels` 的长度不超过预设的最大输入长度 (`max_input_length`) 和最大输出长度 (`max_output_length`)。如果序列太长，将被截断到最大长度。

5. **返回处理后的批次**：最后，函数为每个处理过的批次构建并返回一个包含 `input_ids` 和 `labels` 的字典。这些处理过的数据随后可以被模型用于训练或评估。

将 `process_batch` 通过 `map` 函数应用于数据集时，`map` 函数会自动处理数据集中的每个批次，调用 `process_batch` 函数进行上述处理，并返回一个新的、处理过的数据集。这个处理过程是为了准备数据，使其适用于特定的序列到序列模型训练，特别是在需要考虑对话结构和角色信息时。这样的处理流程是生成任务（如基于对话的系统）的典型准备步骤，它使模型能够根据对话上下文生成或预测下一个合适的回应。
