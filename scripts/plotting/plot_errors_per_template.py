import ast
import os
import sys
from typing import cast

import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.curdir)
sys.path.insert(0, project_root)

import pandas as pd 

from scripts.evals.calculate_all_metrics import full_tools_list 
from src.evals.utils import calculate_metrics, get_latest_results_path 

RESULTS_ROOT_DIR = "data/results/"
MODEL = "gpt-4"

precentage_correct = []
for tool in full_tools_list:
    results = get_latest_results_path(RESULTS_ROOT_DIR, MODEL, tool)
    if results is None:
        continue
    model_results_path, ground_truth_path = results
    predictions = pd.read_csv(model_results_path, dtype=str)
    ground_truth = pd.read_csv(ground_truth_path, dtype=str)
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    predictions["function_calls"] = predictions["function_calls"].apply(ast.literal_eval)
    df = calculate_metrics(ground_truth, predictions, print_errors=False)
    grouped = df.groupby("base_template")["correct"].mean()
    mean_values: list[float] = cast(pd.Series, grouped).values.tolist()
    precentage_correct.append([v * 100 for v in mean_values])
    # print base template with 0% correct
    templates_with_0_percent_correct = (
        df.groupby("base_template")["correct"].mean().loc[df.groupby("base_template")["correct"].mean() == 0]
    )
    print(f"Tool: {tool}")
    print("Base templates with 0% correct:")
    for template in templates_with_0_percent_correct.index:
        print(template)

# flatten
precentage_correct = [item for sublist in precentage_correct for item in sublist]

# print number of template where percentage correct is 100 or 0, and how many are not either 100 or 0
print(
    f"Number of templates where percentage correct is 100 or 0: {precentage_correct.count(100) + precentage_correct.count(0)} out of {len(precentage_correct)} ({(precentage_correct.count(100) + precentage_correct.count(0)) / len(precentage_correct) * 100:.1f}%)"
)
print(
    f"Number of templates where percentage correct is neither 100 or 0: {len(precentage_correct) - precentage_correct.count(100) - precentage_correct.count(0)} out of {len(precentage_correct)} ({(len(precentage_correct) - precentage_correct.count(100) - precentage_correct.count(0)) / len(precentage_correct) * 100:.1f}%)"
)

# Group percentage_correct by value and count
percentage_correct_df = pd.DataFrame(precentage_correct, columns=["percentage_correct"])
percentage_correct_df["count"] = 1
percentage_correct_df = percentage_correct_df.groupby("percentage_correct").count().reset_index()


# increase fontsize
sns.set(font_scale=1.8)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x="percentage_correct", y="count", data=percentage_correct_df)
ax.set(xlabel="Percentage tasks completed correctly", ylabel="Number of templates")
ax.set_xticklabels([f"{x}%" for x in range(0, 200, 10)])
plt.tight_layout()


plt.savefig("data/plots/percentage_correct_per_template.png")
