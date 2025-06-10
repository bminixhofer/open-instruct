from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import List
import yaml
from pathlib import Path
import copy
from itertools import combinations

def get_method_config(args, method, base_model, fts):
    if ":" in method:
        method_name, method_config = method.split(":")
    else:
        method_name = method
        method_config = None

    config = {
        "base_model": base_model,
        "merge_method": method_name,
        "tokenizer": {
            "source": "union",
            "pad_to_multiple_of": args.pad_to_multiple_of,
        },
        "dtype": args.dtype,
        "parameters": {
            "normalize": True,
        }
    }

    if method_name == "linear":
        del config["base_model"]
        model_params = {
            "weight": 1.0
        }
    elif method_name == "ties":
        model_params = {
            "weight": 1.0,
            "density": float(method_config),
        }
    elif method_name == "dare_ties":
        model_params = {
            "weight": 1.0,
            "density": float(method_config),
        }
    elif method_name == "della":
        model_params = {
            "weight": 1.0,
            "density": float(method_config.split(",")[0]),
            "epsilon": float(method_config.split(",")[1]),
        }
    
    config["models"] = [
        {"model": ft, "parameters": copy.deepcopy(model_params)} for ft in fts
    ]
    
    return config

@dataclass
class Args:
    output_dir: str = "automerging_configs"
    base_model: str = "Qwen/Qwen2.5-3B"
    pad_to_multiple_of: int = 64
    dtype: str = "bfloat16"
    fts: List[str] = field(
        default_factory=lambda: [
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-Personas-Math-MATH",
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-Personas-Algebra",
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-Personas-Math-GSM",
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-Personas-Code",
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-Personas-Instruction-Following",
            "MergeMerge/Qwen2.5-3B-Tulu3-SFT-CodeAlpaca",
        ]
    )
    methods: List[str] = field(
        default_factory=lambda: [
            "linear",
            "ties:0.2",
            "ties:0.5",
            "dare_ties:0.2",
            "dare_ties:0.5",
            "della:0.3,0.15",
            "della:0.1,0.05",
        ]
    )

if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        base_config = get_method_config(args, method, args.base_model, args.fts)
        with open(output_dir / f"{method}.yaml", "w") as f:
            yaml.dump(base_config, f)

        for i in range(len(args.fts)):
            path_suffix = f"_minus_{args.fts[i].split('/')[1]}"
            current_fts = args.fts[:i] + args.fts[i+1:]
            config = get_method_config(args, method, args.fts[i], current_fts)
            with open(output_dir / f"{method}{path_suffix}.yaml", "w") as f:
                yaml.dump(config, f)

        for combo in combinations(args.fts, 3):
            model_names = [combo[0].split('/')[1]]

            for ft in combo[1:]:
                name_piece = ft.split('/')[1]
                name_piece = name_piece[name_piece.find("SFT"):]
                model_names.append(name_piece)
            model_names = '_'.join(model_names)
            path_suffix = f"_combo_{model_names}"
            config = get_method_config(args, method, args.base_model, list(combo))
            with open(output_dir / f"{method}{path_suffix}.yaml", "w") as f:
                yaml.dump(config, f)