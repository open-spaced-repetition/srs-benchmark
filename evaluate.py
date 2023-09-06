import pathlib
import json
import numpy as np

if __name__ == "__main__":
    for model in ('FSRSv4', 'FSRSv3', 'LSTM'):
        print(f"Model: {model}")
        m = []
        sizes = []
        result_dir = pathlib.Path(f"./result/{model}")
        result_files = result_dir.glob("*.json")
        for result_file in result_files:
            with open(result_file, "r") as f:
                result = json.load(f)
                m.append(result[model])
                sizes.append(result["size"])

        print(f"Total number of users: {len(sizes)}")
        sizes = np.array(sizes)
        print(f"Total number of reviews: {sizes.sum()}")
        for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
            print(f"metric: {metric}")

            FSRSv4_metrics = np.array([item[metric] for item in m])

            print(f"{model} mean: {np.average(FSRSv4_metrics, weights=sizes):.4f}")

