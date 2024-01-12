import pathlib
import json
import numpy as np
import scipy

dict_x_w = None
            
def weighted_mean(z, axis):
    # creating an array of weights, by mapping z to dict_x_w
    data = np.vectorize(dict_x_w.get)(z)
    return np.average(data[0], weights=data[1], axis=axis)

if __name__ == "__main__":
    for model in (
        "FSRS-4.5",
        "FSRS-rs",
        "FSRSv4",
        "FSRS-4.5-dry-run",
        "SM2",
        "FSRSv3",
        "LSTM",
        "HLR",
    ):
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
        for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
            print(f"metric: {metric}")

            values = np.array([item[metric] for item in m])
            identifiers = [i for i in range(len(values))]
            dict_x_w = {identifier: (value, weight) for identifier, (value, weight) in enumerate(zip(values, sizes))}
            CI_99_bootstrap = scipy.stats.bootstrap((identifiers,), statistic=weighted_mean, confidence_level=0.99, axis=0, method='BCa')
            low = list(CI_99_bootstrap.confidence_interval)[0]
            high = list(CI_99_bootstrap.confidence_interval)[1]
            print(f"{model} mean (n_reviews): {np.average(values, weights=sizes):.4f}")
            print(f"99% CI of {metric} (n_reviews), bootstrapping (scipy): {(high - low) / 2:.4f}")
            dict_x_w = {identifier: (value, np.log(weight)) for identifier, (value, weight) in enumerate(zip(values, sizes))}
            CI_99_bootstrap = scipy.stats.bootstrap((identifiers,), statistic=weighted_mean, confidence_level=0.99, axis=0, method='BCa')
            low = list(CI_99_bootstrap.confidence_interval)[0]
            high = list(CI_99_bootstrap.confidence_interval)[1]
            print(f"{model} mean (ln(n_reviews)): {np.average(values, weights=np.log(sizes)):.4f}")
            print(f"99% CI of {metric} (ln(n_reviews)), bootstrapping (scipy): {(high - low) / 2:.4f}")

        try:
            rmse_bin_again = np.array([item["RMSE(bins)Ratings"]["1"] for item in m])
            print(
                f"{model} mean (RMSE(bins)Ratings[again]): {np.average(rmse_bin_again):.4f}"
            )
            # FSRSv4 mean (RMSE(bins)Ratings[again]): 0.0983
        except KeyError:
            pass

        if len(weights) > 0:
            print(f"weights: {np.median(weights, axis=0).round(4).tolist()}")
