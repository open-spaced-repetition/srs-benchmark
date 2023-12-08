import matplotlib.pyplot as plt
import numpy as np
import json
import pathlib

if __name__ == "__main__":
    model = "FSRS-rs"
    result_dir = pathlib.Path(f"./result/{model}")
    result_files = result_dir.glob("*.json")
    weights = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            weights.append(result["weights"])
    
    weights = np.array(weights)
    print(weights.shape)
    pathlib.Path("./plots").mkdir(parents=True, exist_ok=True)
    for i in range(17):
        plt.hist(weights[:, i], bins=100, log=True)
        median = np.median(weights[:, i])
        mean = np.mean(weights[:, i])
        plt.axvline(median, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
        plt.xlabel('Weight')
        plt.ylabel('Frequency (log scale)')
        plt.legend()
        plt.title(f'w[{i}]')
        plt.savefig(f"./plots/w[{i}].png")
        plt.clf()
