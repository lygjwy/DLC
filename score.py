'''
Learning Complexity Scoring Func for Instruction
'''
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.utils.prune as prune

from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# load fully
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, return_dict=True, load_in_8bit=True, device_map={"": Accelerator().process_index}, low_cpu_mem_usage=True
    )

    return model


def cal_cond_loss(dataset, model, tokenizer):
    # conditional loss
    cls = torch.zeros([0], requires_grad=False, dtype=torch.long).cuda()

    for example in tqdm(dataset):

        if 'instruction' in example.keys():
            # alpaca-cleaned
            instruction = example["instruction"]
            input_text = example["input"]
            response = example["output"]
        elif 'prompt' in example.keys():
            # dolly-hhrlhf
            instruction = example['prompt']
            input_text = ""
            response = example['response']
        elif 'conversations' in example.keys():
            # evol-instruct
            instruction = example['conversations'][0]['value']
            input_text = ""
            response = example['conversations'][1]['value']
        else:
            raise Exception('UNSUPPORTED FORMAT')

        if len(input_text) >= 2:
            input = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:

            '''
            output = f'''{response}'''
        else:
            input = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            '''
            output = f'''{response}'''

        full_idxs = tokenizer.encode(input+output, return_tensors="pt").cuda()
        labels = full_idxs.clone()

        start_token = len(tokenizer.encode(input))
        end_token = full_idxs.shape[1]

        with torch.no_grad():
            outputs = model(full_idxs, labels=labels)

        token_losses = []
        token_logits = outputs.logits
        for i in range(start_token, end_token):
            log_prob_dist = torch.nn.functional.log_softmax(token_logits[0, i-1])

            true_token = full_idxs[0, i]
            token_loss = torch.nn.functional.nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
            token_losses.append(token_loss.item())
        
        if cls.size()[0] == 0:
            cls = torch.mean(torch.asarray(token_losses)).cuda().reshape(1)
        else:
            cls = torch.concatenate((cls, torch.mean(torch.asarray(token_losses)).cuda().reshape(1)), dim=0)

    return cls


def select_idxs(cl, budget_ratio, split_ratio):

    idxs_target = np.array([], dtype=np.int64)
    num = len(cl)

    if split_ratio <= 0.5:
        easy_ratio = min(split_ratio, 0.5 * budget_ratio)
        hard_ratio = budget_ratio - easy_ratio
    else:
        hard_ratio = min(1-split_ratio, 0.5 * budget_ratio)
        easy_ratio = budget_ratio - hard_ratio
    
    # easy part
    idxs_target = np.append(idxs_target, np.random.choice(np.argsort(cl)[:round(num * split_ratio)], size=round(num * easy_ratio), replace=False))

    # hard part
    idxs_target = np.append(idxs_target, np.random.choice(np.argsort(cl)[round(num * split_ratio):], size=round(num * hard_ratio), replace=False))

    return idxs_target


def main(args):

    init_seed(args.seed)
    inter = args.interval
    freq = args.frequency

    # load the dataset
    train_dataset = load_dataset(
        args.dataset,
        split='train',
        token=True,
        num_proc=8
    )

    # load the ckpt model (with PEFT)
    model = load_model(args.model_id)
    # model = load_peft_model(model, args.peft_model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    t1 = time.time()
    cl_list = []

    for _ in range(freq):

        cl = cal_cond_loss(train_dataset, model, tokenizer)
        cl_list.append(cl.cpu().numpy())

        for _, module in model.named_modules():
            # parameters_to_prune = []
            if isinstance(module, torch.nn.Linear):
                # parameters_to_prune.append((module, 'weight'))
                prune.global_unstructured([(module, 'weight')], pruning_method=prune.L1Unstructured, amount=inter)
                prune.remove(module, 'weight')
                torch.cuda.empty_cache()

    cls = np.stack(cl_list, axis=0)
    cl = np.average(cls, axis=0)

    for split in [0.25, 0.75]:

        idxs_dir = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset.split('/')[1], args.model_id.split('/')[1], 'lwot-l1-s%s-i%s_f%s' % (str(split), str(inter), str(freq)))
        os.makedirs(idxs_dir, exist_ok=True)

        np.save(os.path.join(idxs_dir, 'cls'), cls)
        np.save(os.path.join(idxs_dir, 'cl'), cl)

        for i in [0.50]:
            budget_ratio = i
            
            idxs = select_idxs(cl, budget_ratio, split)

            np.save(os.path.join(idxs_dir, str(budget_ratio)), idxs)

    print('Time (%s - %s - %ss): ' % (args.dataset, args.model_id, str(time.time() - t1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pruning Score')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", choices=["yahma/alpaca-cleaned", "MaziyarPanahi/WizardLM_evol_instruct_V2_196k", "mosaicml/dolly_hhrlhf"])
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-8B", choices=["mistralai/Mistral-7B-v0.3", "meta-llama/Meta-Llama-3-8B", "google/gemma-2-9b"])

    # parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--frequency', type=int, default=3)

    args = parser.parse_args()
    main(args)