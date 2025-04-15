from glob import glob
import pandas as pd

if __name__ == "__main__":
    scores = []
    header = ["Model", "GSM8K", "MATH", "HumanEval", "MBPP", "IFEval"]

    rows = []

    for model_path in sorted(glob("logs/**/*")):
        if not "automerging_outputs" in model_path or "della" in model_path:
            continue

        print(model_path)
        gsm_score = float(open(f"{model_path}/gsm.log").readlines()[-1].split(":")[-1].strip())
        print(gsm_score)
        try:
            humaneval_score = float(open(f"{model_path}/humaneval.log").readlines()[-1].split(":")[-1].strip().rstrip("}"))
        except:
            humaneval_score = None
        print(humaneval_score)
        try:
            mbpp_score = float(open(f"{model_path}/mbpp.log").readlines()[-1].split(":")[-1].strip().rstrip("}"))
        except:
            mbpp_score = None
        print(mbpp_score)

        for line in open(f"{model_path}/MATH.log").readlines():
            if "Accuracy:" in line:
                math_score = float(line.split(":")[-1].strip())
                break

        print(math_score)

        try:
            ifeval_lines = open(f"{model_path}/ifeval.log").readlines()
            for i, line in enumerate(ifeval_lines):
                if "Running loose evaluation..." in line:
                    ifeval_score = float(ifeval_lines[i + 1].split(":")[-1].strip()[3:])
                    break
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
    with pd.option_context('display.max_colwidth', 100):
        print(df)