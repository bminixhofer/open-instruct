from glob import glob
import json
import pandas as pd

if __name__ == "__main__":
    scores = []
    header = ["Model", "GSM8K", "MATH", "HumanEval", "MBPP", "IFEval"]

    rows = []

    for model_path in sorted(glob("logs/automerging_outputs_combo/*")):
        if not "automerging_outputs" in model_path or "della" in model_path:
            continue

        print(model_path)
        gsm_score = json.load(open(f"{model_path}/gsm/metrics.json"))["exact_match"]
        print(gsm_score)

        try:
            humaneval_score = json.load(open(f"{model_path}/humaneval/metrics.json"))["pass@1"]
        except:
            humaneval_score = None
        print(humaneval_score)

        try:
            mbpp_score = json.load(open(f"{model_path}/mbpp/metrics.json"))["pass@1"]
        except:
            mbpp_score = None
        print(mbpp_score)

        try:
            math_score = json.load(open(f"{model_path}/MATH/metrics.json"))["accuracy"]
        except:
            math_score = None

        print(math_score)

        try:
            ifeval_score = json.load(open(f"{model_path}/ifeval/metrics.json"))["loose"]["Accuracy"]
        except:
            ifeval_score = None

        rows.append((
            model_path[len("logs/"):],
            gsm_score,
            math_score,
            humaneval_score,
            mbpp_score,
            ifeval_score
        ))

    df = pd.DataFrame(rows, columns=header)
    # high max colwidth
    with pd.option_context('display.max_colwidth', 200, 'display.max_columns', 10, 'display.max_rows', 200):
        print(df)