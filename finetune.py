import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="bigcode/starcoder")
    parser.add_argument('--dataset_name', type=str, default="bigcode/the-stack-dedup")
    parser.add_argument('--subset', type=str, default="data")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--size_valid_set', type=int, default=4000)
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--shuffle_buffer', type=int, default=5000)

    parser.add_argument('--data_column', type=str, default="content")

    parser.add_argument('--seq_length', type=int, default=2048)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=10)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default="./checkpoints")
    parser.add_argument('--log_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--save_freq', default=500, type=int)

    # Lora config
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=0)
    parser.add_argument('--lora_dropout', type=float, default=0)
    parser.add_argument('--lora_target_modules', type=str, default=None)

    parser.add_argument('--use_flash_attention', action='store_true')

    parser.add_argument('--use_8bit_optimizer', action='store_true')

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in zip(range(nb_examples), dataset):
        text = example[data_column]
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field='content'
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)['input_ids']
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)


def setup_model(model_path, use_lora=False, lora_r=0, lora_alpha=0, lora_dropout=0, lora_target_modules=None):
    """
    Setup model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

    if use_lora:
        assert lora_r > 0, "LoRA r must be > 0"
        assert lora_alpha > 0, "LoRA alpha must be > 0"
        config = AutoConfig.from_pretrained(model_path, use_auth_token=True)

        # We need to use bfloat16 to use int8
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_auth_token=True
        )

        model = prepare_model_for_int8_training(model)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules.split(',') if lora_target_modules else None,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_auth_token=True
        )

    return model, tokenizer


def create_datasets(tokenizer, args):
    """
    Create datasets for trainer
    """
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

    if args.streaming:
        print("Loading dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=args.size_valid_set, seed=args.seed, shuffle=True)
        train_data = dataset['train']
        valid_data = dataset['test']
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    """
    Run training with the given arguments
    """
    if args.use_lora:
        args.gradient_checkpointing = True

    print("Loading model and tokenizer")
    model, tokenizer = setup_model(
        args.model_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules
    )

    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="starcoder-finetune",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.save_model()


def main():
    args = parse_args()

    set_seed(args.seed)

    # Set RoPE scaling factor
    # os.environ["ROPE_SCALING"] = "dynamic"

    accelerator = Accelerator()

    model, tokenizer = setup_model(
        args.model_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules
    )

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()