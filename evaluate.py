import pathlib
import json
import numpy as np

if __name__ == "__main__":
    for model in ('FSRS-rs', 'FSRSv4', 'FSRSv3', 'LSTM', 'HLR', 'SM2'):
        print(f"Model: {model}")
        m = []
        weights = []
        sizes = []
        result_dir = pathlib.Path(f"./result/{model}")
        result_files = result_dir.glob("*.json")
        for result_file in result_files:
            with open(result_file, "r") as f:
                result = json.load(f)
                m.append(result[model])
                sizes.append(result["size"])
                if "weights" in result:
                    weights.append(result["weights"])

        print(f"Total number of users: {len(sizes)}")
        sizes = np.array(sizes)
        print(f"Total number of reviews: {sizes.sum()}")
        sizes = np.log(sizes)
        for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
            print(f"metric: {metric}")

            FSRSv4_metrics = np.array([item[metric] for item in m])

            print(f"{model} mean: {np.average(FSRSv4_metrics, weights=sizes):.4f}")

        if len(weights) > 0:
            print(f"weights: {np.median(weights, axis=0).round(4).tolist()}")