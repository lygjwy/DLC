'''
Learning Complexity Scoring Func for Instruction
'''
import os
import time
import random
import argparse
import numpy as np

import torch

from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

from algorithms import get_data_pruner


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# load fully
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, return_dict=True, load_in_8bit=False, device_map={"": Accelerator().process_index}, low_cpu_mem_usage=True
    )

    return model


def main(args):

    init_seed(args.seed)
    idxs_dir = os.path.join(args.idxs_dir, 'seed_'+str(args.seed), args.dataset.split('/')[1], args.model_id.split('/')[1], args.principle)
    os.makedirs(idxs_dir, exist_ok=True)

    # load the dataset
    train_dataset = load_dataset(
        args.dataset,
        split='train',
        token=True,
        num_proc=8
    )

    # load the ckpt model
    model = load_model(args.model_id)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    t1 = time.time()
    pruner = get_data_pruner(args.principle, train_dataset, model, tokenizer)

    # save to local
    # for i in [.50]:
    i = args.ratio
    idxs_list = pruner.prune(i)
    storage_path = os.path.join(idxs_dir, str(i))
    np.save(storage_path, idxs_list)
    
    print('Time (%s - %s - %ss): ' % (args.dataset.split('/')[1], args.model_id.split('/')[1], str(time.time() - t1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--idxs_dir', type=str, default='./idxs')
    parser.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", choices=["yahma/alpaca-cleaned", "MaziyarPanahi/WizardLM_evol_instruct_V2_196k", "mosaicml/dolly_hhrlhf"])
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-8B", choices=["mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-2-9b", "google/gemma-2-9b-it"])

    parser.add_argument('--principle', type=str, default='random', choices=['random'])
    parser.add_argument('--ratio', type=float, default=0.5) # preserving ratio

    args = parser.parse_args()
    main(args)