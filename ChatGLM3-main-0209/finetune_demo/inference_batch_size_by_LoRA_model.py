from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from finetune_hf import *

def load_tokenizer_and_model(model_directory = "/home/bingxing2/home/scx6mdq/myllm/ChatGLM3-main-0209/finetune_demo/output-type2-MDAtrain/checkpoint-33000" ):
    # 加载经过 PEFT 微调的模型
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_directory, trust_remote_code=True, device_map='auto'
    )

    # 获取分词器的目录
    tokenizer_dir = model.peft_config['default'].base_model_name_or_path

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return tokenizer, model
def _prepare_model_for_training(model: nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

def save_predictions(predictions, file_path):
    """将预测结果保存到文件"""
    with open(file_path, "a") as f:  # 使用追加模式以防止覆盖之前的内容
        for pred in predictions:
            f.write(pred + "\n")


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir)
    data_manager = DataManager(data_dir, ft_config.data_config)

    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    # _sanity_check(
    #     train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    # )

    # turn model to fp32
    _prepare_model_for_training(model)

    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )
    # trainer.train()

    # test stage
    predictions = trainer.predict(test_dataset)
    save_predictions(predictions, "test_predictions.txt")


if __name__ == '__main__':
    app()
