import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable
import yaml
import numpy as np
from pathlib import Path
from mergekit.merge import run_merge
from mergekit.config import MergeConfiguration
from mergekit.options import MergeOptions

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser


def _as_flattened(s):
    if isinstance(s, Iterable):  # tensor or tensor sequence
        return torch.cat([i.flatten() for i in tqdm(s)])
    else:  # scalar
        return s
    

@dataclass
class Args:
    output: str = "diffs"
    prefix: str = "automerging_configs/"
    suffix: str = "_minus_Qwen2.5-3B-Tulu3-SFT-CodeAlpaca"
    tmp_cache_dir: str = "automerging_outputs/"
    merges: list[str] = field(default_factory=lambda: ["ties:0.2", "ties:0.5", "linear", "dare_ties:0.2", "dare_ties:0.5"])
    base_model: str | None = None

if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()

    if args.base_model is not None:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).eval()
    else:
        base_model = None

    all_flat_params = []

    merge_options = MergeOptions(cuda=True)

    for merge_infix in args.merges:
        merge_config = args.prefix + merge_infix + args.suffix + ".yaml"
        model_path = args.tmp_cache_dir + merge_infix + args.suffix
        if not os.path.exists(merge_config):
            raise FileNotFoundError(f"Merge configuration file {merge_config} does not exist.")
        
        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            config_source = open(merge_config, "r", encoding="utf-8").read()
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(config_source))

            run_merge(
                merge_config,
                model_path,
                options=merge_options,
                config_source=config_source,
            )
            model = AutoModelForCausalLM.from_pretrained(model_path)

        flat_params = _as_flattened(model.state_dict().values())
        all_flat_params.append(flat_params)

    distances_l1 = np.zeros((len(all_flat_params), len(all_flat_params)))
    distances_l2 = np.zeros((len(all_flat_params), len(all_flat_params)))
    for i in tqdm(range(len(all_flat_params))):
        for j in range(i + 1, len(all_flat_params)):
            distances_l1[i, j] = torch.dist(all_flat_params[i], all_flat_params[j], p=1).item()
            distances_l1[j, i] = distances_l1[i, j]            
            distances_l2[i, j] = torch.dist(all_flat_params[i], all_flat_params[j], p=2).item()
            distances_l2[j, i] = distances_l2[i, j]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / ("distances_l1" + args.suffix + ".npy"), distances_l1)
    np.save(output / ("distances_l2" + args.suffix + ".npy"), distances_l2)