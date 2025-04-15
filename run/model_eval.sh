# to prepare the data: python3 scripts/data/prepare_eval_data.sh

touch eval/__init__.py

MODEL_NAME=$1

mkdir -p "logs/${MODEL_NAME}"

echo "Evaluating model: $MODEL_NAME"

time ipython eval/gsm/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_dir=data/eval/gsm \
    --save_dir=logs/${MODEL_NAME}/gsm \
    --use_chat_format \
    --additional_stop_sequence $'\n\nQuestion:' \
    --use_vllm 2>&1 | tee -a logs/${MODEL_NAME}/gsm.log

time ipython eval/MATH/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_dir=data/eval/MATH \
    --save_dir=logs/${MODEL_NAME}/MATH \
    --use_chat_format \
    --use_vllm 2>&1 | tee -a logs/${MODEL_NAME}/MATH.log

time ipython eval/codex_humaneval/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --data_file=data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --data_file_hep=data/eval/codex_humaneval/humanevalpack.jsonl \
    --save_dir=logs/${MODEL_NAME}/humaneval \
    --unbiased_sampling_size_n=1 \
    --use_chat_format \
    --use_vllm 2>&1 | tee -a logs/${MODEL_NAME}/humaneval.log

HF_ALLOW_CODE_EVAL=1 time ipython eval/mbpp/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --use_chat_format \
    --chat_formatting_function=eval.templates.create_prompt_with_tulu_chat_format \
    --save_dir=logs/${MODEL_NAME}/mbpp \
    --unbiased_sampling_size_n=1 \
    --use_vllm 2>&1 | tee -a logs/${MODEL_NAME}/mbpp.log
cat logs/${MODEL_NAME}/mbpp/metrics.json >> logs/${MODEL_NAME}/mbpp.log

time ipython eval/ifeval/run_eval.py -- \
    --model_name_or_path=$MODEL_NAME \
    --save_dir=logs/${MODEL_NAME}/ifeval \
    --use_chat_format \
    --use_vllm 2>&1 | tee -a logs/${MODEL_NAME}/ifeval.log