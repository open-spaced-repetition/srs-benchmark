import pathlib
import json
import numpy as np
import scipy
import math

dict_x_w = None


def weighted_mean(z, axis):
    # creating an array of weights, by mapping z to dict_x_w
    data = np.vectorize(dict_x_w.get)(z)
    return np.average(data[0], weights=data[1], axis=axis)

def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    def first_nonzero_digit(x):
        x = str(x)
        for digit in x:
            if digit == "0" or digit == ".":
                pass
            else:
                return int(digit)

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = min(len(str(CI)[2 + n_lead_zeros_CI:]), 2)
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    first_sigdig_CI = first_nonzero_digit(rounded_CI)
    if first_sigdig_CI<5:
        rounded_value = round(value, decimals - 1)
        return str(f'{rounded_value:.{decimals - 1}f}'), str(f'{rounded_CI:.{decimals}f}')
    else:
        rounded_value = round(value, max(decimals - 2, 0))
        rounded_CI = round(CI, max(decimals - 1, 1))
        return str(f'{rounded_value:.{max(decimals - 2, 0)}f}'), str(f'{rounded_CI:.{max(decimals - 1, 1)}f}')


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
            CI_99_bootstrap = scipy.stats.bootstrap((identifiers,), statistic=weighted_mean, confidence_level=0.99,
                                                    axis=0, method='BCa')
            low = list(CI_99_bootstrap.confidence_interval)[0]
            high = list(CI_99_bootstrap.confidence_interval)[1]
            CI = (high - low) / 2
            wmean = np.average(values, weights=sizes)
            rounded_mean, rounded_CI = sigdig(wmean, CI)
            # asymm = (high - wmean)/(wmean - low)
            print(f"{model} mean (n_reviews): {rounded_mean}")
            print(f"99% CI of {metric} (n_reviews), bootstrapping (scipy): {rounded_CI}")
            # print(f"Asymmetry of the CI={asymm:.4f}")
            dict_x_w = {identifier: (value, np.log(weight)) for identifier, (value, weight) in
                        enumerate(zip(values, sizes))}
            CI_99_bootstrap = scipy.stats.bootstrap((identifiers,), statistic=weighted_mean, confidence_level=0.99,
                                                    axis=0, method='BCa')
            low2 = list(CI_99_bootstrap.confidence_interval)[0]
            high2 = list(CI_99_bootstrap.confidence_interval)[1]
            CI2 = (high2 - low2) / 2
            wmean2 = np.average(values, weights=np.log(sizes))
            rounded_mean2, rounded_CI2 = sigdig(wmean2, CI2)
            # asymm2 = (high2 - wmean2)/(wmean2 - low2)
            print(f"{model} mean log(n_reviews): {rounded_mean2}")
            print(f"99% CI of {metric} log(n_reviews), bootstrapping (scipy): {rounded_CI2}")
            # print(f"Asymmetry of the CI={asymm2:.4f}")

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
        print('')
