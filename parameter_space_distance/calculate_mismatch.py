import argparse
import os
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

def _as_flattened(s):
    if isinstance(s, Iterable):  # tensor or tensor sequence
        return torch.cat([i.flatten() for i in tqdm(s)])
    else:  # scalar
        return s

def run(args):

    actual_vocab_sizes = {"qwen2.5": 151645} # https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/tokenizer_config.json

    multitask_model = os.path.join(args.multitask)
    multitask_model = AutoModelForCausalLM.from_pretrained(multitask_model).to("cuda:0")

    finetuned_model = os.path.join(args.finetuned)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model).to("cuda:0")

    if args.model_family not in actual_vocab_sizes.keys():
        raise ValueError(f"{args.model_family} is not supported. Please add its actual vocabulary size.")
    else:
        vocab_size = actual_vocab_sizes[args.model_family]

    with torch.no_grad():
        finetune = [{"name": name, "param": p} for name, p in finetuned_model.named_parameters()]
        multitask = [{"name": name, "param": p} for name, p in multitask_model.named_parameters()]

        finetune_params = []
        multitask_params = []

        for idx in range(len(finetune)):

            if finetune[idx]["name"] == "model.embed_tokens.weight":

                finetune_params.append(finetune[idx]["param"][:vocab_size])
                multitask_params.append(multitask[idx]["param"][:vocab_size])
            else:
                finetune_params.append(finetune[idx]["param"])
                multitask_params.append(multitask[idx]["param"])

        finetuned_model = _as_flattened(finetune_params)
        multitask_model = _as_flattened(multitask_params)

        results = {}
        for p in [1,2]:
            mismatch = torch.norm(finetuned_model - multitask_model, p = p) * (1/len(finetuned_model) * 1e8)
            mismatch = mismatch.detach().cpu().numpy().item()
            print(f"mismatch p={p}: {mismatch}")
            results[p] = mismatch

        del multitask_model
        del finetuned_model

        del finetune_params
        del multitask_params

        del finetune
        del multitask

        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetuned",
        type=str,
    )

    parser.add_argument(
        "--multitask",
        type=str,
    )

    parser.add_argument(
        "--model_family",
        type=str,
        default="qwen2.5"
    )

    parsed_args = parser.parse_args()

    run(parsed_args)