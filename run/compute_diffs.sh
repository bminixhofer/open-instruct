python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-Personas-Code"
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-Personas-Instruction-Following"
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix=""
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-Personas-Math-GSM"
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-CodeAlpaca"
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-Personas-Algebra"
rm -r automerging_outputs/*

python3 run/compute_parameter_diff.py --suffix="_minus_Qwen2.5-3B-Tulu3-SFT-Personas-Math-MATH"
rm -r automerging_outputs/*
