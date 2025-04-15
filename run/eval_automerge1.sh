#!/bin/bash
#SBATCH -J eval_automerge1
#SBATCH --time=12:0:0
#SBATCH -wltl-gpu05
#SBATCH -G1

for config in automerging_configs/linear*.yaml; do
    mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
    bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
    rm -rf automerging_outputs/$(basename $config .yaml)
done

for config in automerging_configs/ties*.yaml; do
    mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
    bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
    rm -rf automerging_outputs/$(basename $config .yaml)
done

for config in automerging_configs/dare_ties:0.5*.yaml; do
    mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
    bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
    rm -rf automerging_outputs/$(basename $config .yaml)
done

# for config in automerging_configs/della*.yaml; do
#     mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
#     bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
#     rm -rf automerging_outputs/$(basename $config .yaml)
# done

# for config in automerging_configs/linear*.yaml; do
#     mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
#     bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
#     rm -rf automerging_outputs/$(basename $config .yaml)
# done

# for config in automerging_configs/ties*.yaml; do
#     mergekit-yaml $config automerging_outputs/$(basename $config .yaml)
#     bash run/model_eval.sh automerging_outputs/$(basename $config .yaml)
#     rm -rf automerging_outputs/$(basename $config .yaml)
# done