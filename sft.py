'''
Supervised Instruction Tuning
'''

import os
import argparse

import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
# from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, logging, set_seed

from trl import SFTTrainer
# from trl.trainer import ConstantLengthDataset


def format_prompts_func(examples):
    
    output_text = []

    if 'instruction' in examples.keys():
        nb = len(examples['instruction'])
    elif 'prompt' in examples.keys():
        nb = len(examples['prompt'])
    elif 'conversations' in examples.keys():
        nb = len(examples['conversations'])
    else:
        raise Exception('UNSUPPORTED KEY')

    for i in range(nb):
        
        if 'instruction' in examples.keys():
            # alpaca-cleaned
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]
        elif 'prompt' in examples.keys():
            # dolly-hhrlhf
            instruction = examples['prompt'][i]
            input_text = ""
            response = examples['response'][i]
        elif 'conversations' in examples.keys():
            # evol-instruct
            instruction = examples['conversations'][i][0]['value']
            input_text = ""
            response = examples['conversations'][i][1]['value']
        else:
            raise Exception('UNSUPPORTED FORMAT')

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        output_text.append(text)

    return output_text


def print_trainable_params(model):
    """
    Print the number of trainable parameters in the model
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )


def run_training(args, train_data, tokenzier):
    print('Loading the model')

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    train_data.start_iteration = 0

    print('Starting main loop')

    training_args = TrainingArguments(
        output_dir=args.ckpts_dir,
        dataloader_drop_last=True,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        # run_name=args.run_name,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenzier,
        max_seq_length=args.seq_length,
        args=training_args,
        train_dataset=train_data,
        peft_config=lora_config,
        formatting_func=format_prompts_func,
        packing=False
    )

    print_trainable_params(trainer.model)

    print("Training...")
    trainer.train()

    return trainer.model


def main(args):

    exp_dir = os.path.join(args.ckpts_dir, 'seed_'+str(args.seed), args.dataset.split('/')[1]+'-'+args.model_id.split('/')[1], args.principle+'-'+str(args.ratio))
    os.makedirs(exp_dir, exist_ok=True)

    train_dataset = load_dataset(
        args.dataset,
        split=args.split,
        token=True,
        num_proc=args.num_workers
    )

    idxs_path = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset.split('/')[1], args.model_id.split('/')[1], args.principle, str(args.ratio)+'.npy')
    idxs = np.load(idxs_path).astype(int)

    train_dataset = train_dataset.select(idxs)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = run_training(args, train_dataset, tokenizer)

    print("Saving last checkpoint of the model")
    model.save_pretrained(exp_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Supervised Fintuning with PEFT')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", choices=["yahma/alpaca-cleaned", "mosaicml/dolly_hhrlhf"]) # "MaziyarPanahi/WizardLM_evol_instruct_V2_196k", "HuggingFaceH4/ultrachat_200k"
    parser.add_argument('--split', type=str, default='train') # w.o.t validation splitting
    parser.add_argument("--ckpts_dir", type=str, default="./ckpts")

    # pruning
    parser.add_argument('--principle', type=str, default='random')
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--ratio', type=float, default=0.5)

    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-8B", choices=["mistralai/Mistral-7B-v0.3", "meta-llama/Meta-Llama-3-8B", "google/gemma-2-9b"])
    
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--save_freq", default=5000, type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    logging.set_verbosity_error()

    main(args)