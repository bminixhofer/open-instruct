import os
import json
import shutil
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_mergekit(config_path, output_path):
    try:
        subprocess.run(['mergekit-yaml', config_path, output_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {config_path}: {e}")


def run_calculate_mismatch(output_path, multitask_path, model_family):
    try:
        from calculate_mismatch import run
        from argparse import Namespace
        results = run(args=Namespace(finetuned=output_path, multitask=multitask_path, model_family=model_family))
        return results

    except subprocess.CalledProcessError as e:
        print(f"Error running calculate_mismatch: {e}")


def download_multitask_model(save_path):
    try:
        model = AutoModelForCausalLM.from_pretrained("MergeMerge/Qwen2.5-3B-Tulu3-SFT-full-mixture")
        tokenizer = AutoTokenizer.from_pretrained("MergeMerge/Qwen2.5-3B-Tulu3-SFT-full-mixture")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        print(f"Error downloading model: {e}")


def do_comparison(merging_configs_path, output_base_path, multitask_path, model_family):
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    if not os.path.exists(multitask_path):
        os.makedirs(multitask_path)
    #download_multitask_model(multitask_path)

    for filename in os.listdir(merging_configs_path):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            config_path = os.path.join(merging_configs_path, filename)
            output_path = os.path.join(output_base_path, os.path.splitext(filename)[0])
            results_file = os.path.join(output_path, "results.json")
            if os.path.exists(results_file):
                print(f"results file {results_file} already exists, skipping")
                continue

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            merged_model_path = os.path.join(output_path, "merged_model")
            if not os.path.exists(merged_model_path):
                print(f"starting merge now!")
                run_mergekit(config_path, merged_model_path)
            print("starting mismatch calculation now!")
            results = run_calculate_mismatch(merged_model_path, multitask_path, model_family=model_family)
            results["filename"] = filename
            results["multitask_path"] = multitask_path
            print(results)

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            if os.path.exists(merged_model_path):
                shutil.rmtree(merged_model_path)

def generate_and_save_summary(output_base_path, ):
    results_summary = {}

    # Cycle through each directory in output path
    for dirname in os.listdir(output_base_path):
        full_results = {}
        merge_settings = dirname.split("_")
        for merging_method in merging_methods:
            if dirname.startswith(merging_method):
                full_results["method"] = merging_method
                models_as_string = dirname.removeprefix(merging_method)[1:]
        merge_settings = models_as_string.split("_")
        # full_results["method"] = merge_settings[0]
        if len(models_as_string) > 0 and len(merge_settings) > 0:
            if merge_settings[0] == "minus":
                for model in models:
                    if model in merge_settings[1]:
                        full_results[model] = False
                    else:
                        full_results[model] = True
            elif merge_settings[0] == "combo":
                for merged_model in merge_settings[1:]:
                    merged_model = merged_model.removeprefix("Qwen2.5-3B-Tulu3-")
                    if merged_model in models:
                        full_results[merged_model] = True
                    else:
                        full_results[merged_model] = False
            else:
                print(f"unknown combination: {merge_settings[0]} in {dirname}")
                continue

        dir_path = os.path.join(output_base_path, dirname)
        if os.path.isdir(dir_path):
            results_file = os.path.join(dir_path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    full_results["L1"] = results["1"]
                    full_results["L2"] = results["2"]
                    # results_summary[dirname] = results
        results_summary[dirname] = full_results

    # Save overall summary
    summary_path = os.path.join(output_base_path, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)

if __name__ == "__main__":
    models = ["SFT-Personas-Math-MATH",
              "SFT-Personas-Algebra",
              "SFT-Personas-Math-GSM",
              "SFT-Personas-Code",
              "SFT-Personas-Instruction-Following",
              "SFT-CodeAlpaca"]

    merging_methods = ["dare_ties:0.2",
                       "dare_ties:0.5",
                       "della:0.1,0.05",
                       "della:0.3,0.15",
                       "linear",
                       "ties:0.2",
                       "ties:0.5"]

    merging_configs_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/model_merging/merging_benchmark/automerging_configs"
    output_base_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/model_merging/merging_benchmark/results/multitask_v_merge_param_space"
    multitask_path = os.path.join(output_base_path, "multitask")

    model_family = "qwen2.5" # is needed for vocab truncation

    do_comparison(merging_configs_path, output_base_path, multitask_path, model_family)

    generate_and_save_summary(output_base_path)



