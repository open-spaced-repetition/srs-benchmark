import pathlib
import json
import numpy as np

if __name__ == "__main__":
    FSRSv4 = []
    sizes = []
    result_dir = pathlib.Path("./result/FSRSv4")
    result_files = result_dir.glob("*.json")
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            FSRSv4.append(result["FSRS v4"])
            sizes.append(result["size"])

    print(f"Total number of users: {len(sizes)}")
    sizes = np.array(sizes)
    print(f"Total size: {sizes.sum()}")
    for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
        print(f"metric: {metric}")

        FSRSv4_metrics = np.array([item[metric] for item in FSRSv4])

        print(f"FSRSv4 mean: {np.average(FSRSv4_metrics, weights=sizes):.4f}")

