# to prepare the data: python3 scripts/data/prepare_eval_data.sh

touch eval/__init__.py

MODEL_NAME=$1

mkdir -p "logs/${MODEL_NAME}"

ipython --pdb eval/gsm/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_dir=data/eval/gsm \
    --use_chat_format \
    --use_vllm | tee -a logs/${MODEL_NAME}/gsm.log

ipython --pdb eval/MATH/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_dir=data/eval/MATH \
    --use_chat_format \
    --use_vllm | tee -a logs/${MODEL_NAME}/MATH.log

ipython --pdb eval/codex_humaneval/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_file=data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --data_file_hep=data/eval/codex_humaneval/humanevalpack.jsonl \
    --use_chat_format \
    --use_vllm | tee -a logs/${MODEL_NAME}/humaneval.log