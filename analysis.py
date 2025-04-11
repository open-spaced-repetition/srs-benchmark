import matplotlib.pyplot as plt
import numpy as np
import json
import pathlib
from KDEpy import FFTKDE  # type: ignore
from fsrs_optimizer import DEFAULT_PARAMETER  # type: ignore


def chen_rule(data, weights=None):
    # https://www.hindawi.com/journals/jps/2015/242683/
    data = np.asarray(data)
    if weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.asarray(weights)

    def weighted_percentile(data, weights, q):
        # q must be between 0 and 1
        ix = np.argsort(data)
        data = data[ix]  # sort data
        weights = weights[ix]  # sort weights
        C = 1
        # 0 = 'weibull'
        # 1/3 = 'median_unbiased'
        # 3/8 = 'normal_unbiased'
        # 1/2 = 'hazen'
        # 1 = 'linear'
        cdf = (np.cumsum(weights) - C * weights) / (
            np.sum(weights) + (1 - 2 * C) * weights
        )  # 'like' a CDF function
        return np.interp(
            q, cdf, data
        )  # when all weights are equal to 1, this is equivalent to using 'linear' in np.percentile

    std = np.sqrt(np.cov(data, aweights=weights))
    IQR = (
        weighted_percentile(data, weights, q=0.75)
        - weighted_percentile(data, weights, q=0.25)
    ) / 1.3489795003921634
    scale = min(IQR, std)
    mean = np.average(data, weights=weights)
    n = len(data)
    if mean != 0 and scale > 0:
        cv = (1 + 1 / (4 * n)) * scale / mean  # corrected for small sample size
        h = ((4 * (2 + cv**2)) ** (1 / 5)) * scale * (n ** (-2 / 5))
        return h
    else:
        raise Exception("Chen's rule failed")


def mode_of_three(data):
    assert len(data) == 3
    data = np.sort(np.asarray(data))
    epsilon = 1e-8
    # this is just to avoid division by 0
    weights = np.ones(3)
    const = 1.2
    if data[1] - data[0] < data[2] - data[1]:
        shortest_distance = np.maximum(data[1] - data[0], epsilon)
        u = (data[2] - data[1]) / shortest_distance
        weights[2] = np.where(u < const, 1, (const / u) ** 2)
        return np.dot(data, weights) / np.sum(weights)
        # distance-weighted average, the furthest datapoint is assigned a low weight if it's far away from the other two
    elif data[1] - data[0] > data[2] - data[1]:
        shortest_distance = np.maximum(data[2] - data[1], epsilon)
        u = (data[1] - data[0]) / shortest_distance
        weights[0] = np.where(u < const, 1, (const / u) ** 2)
        return np.dot(data, weights) / np.sum(weights)
        # distance-weighted average, the furthest datapoint is assigned a low weight if it's far away from the other two
    else:
        return data[1]


def HSM(a):
    array = np.sort(np.asarray(a))

    def iteration(a):
        j = -1
        w_min = a[-1] - a[0]
        n = len(a)
        N = (n - 1) // 2 + 1

        for i in range(n - N):
            w = a[i + N - 1] - a[i]
            if w <= w_min:
                w_min = w
                j = i

        return a[j : j + N]

    while True:
        if array[-1] == array[0]:
            # it doesn't matter which value is returned in this case
            return array[0]
        elif len(array) == 1 or len(array) == 2:
            return np.mean(array)
        elif len(array) == 3:
            return mode_of_three(array)
        else:
            array = iteration(array)


# this one is very slow
def HRM(v):
    # https://sci-hub.se/10.1016/S0167-9473(01)00057-3
    # https://github.com/kfarr3/Half-Range-Mode-Estimation/blob/master/Half%20Range%20Mode%20Estimation.ipynb
    v = np.sort(np.asarray(v))

    def iteration(v):
        N = len(v)
        # calculate the interval width, this method gets it's name
        # with a Beta of 0.5 or half-width.  Other Beta values can
        # be used for different effects
        # This is half the width of the full range of data
        w = 0.5 * (v[-1] - v[0])

        # Create N-1 intervals called I
        # each interval is of w width
        I = []
        for j in range(0, N - 1):  # j = 1 to N-1, paper is 1 based index
            I.append((v[j], v[j] + w))
        I = np.array(I)

        # for each interval, determine how many values are in each interval
        cnt = np.array([((rng[0] <= v) & (v <= rng[1])).sum() for rng in I])
        N_prime = max(cnt)

        if (cnt == N_prime).sum() == 1:
            J = I[np.where(cnt == N_prime)[0][0]]
            v = v[np.logical_and(v >= J[0], v <= J[1])]
            return v

        IJ = []
        for Ii in I[cnt == N_prime]:
            IJ.append(v[(Ii[0] <= v) & (v <= Ii[1])])

        w_prime = np.ptp(IJ, axis=1).min()

        Vmin = v[-1]  # default to our array's min/max
        Vmax = v[0]
        for IJi in IJ:
            if (IJi[-1] - IJi[0]) == w_prime:
                if IJi[0] < Vmin:
                    Vmin = IJi[0]
                if IJi[-1] > Vmax:
                    Vmax = IJi[-1]

        min_index = np.argmax(v == Vmin)
        v_back = v[::-1]
        max_index = len(v) - np.argmax(v_back == Vmax) - 1
        N_prime_prime = max_index - min_index + 1

        v = v[min_index : max_index + 1]

        if N == N_prime_prime:
            # this should not happen for continous data, but regardless we need to have a case for it
            # Essentially this means that we did not progress this itteration
            if (v[2] - v[1]) < (v[-1] - v[-2]):
                v = v[:-1]
            elif (v[2] - v[1]) > (v[-1] - v[-2]):
                v = v[1:]
            else:
                v = v[1:-1]

        return v

    while True:
        if v[-1] == v[0]:
            # it doesn't matter which value is returned in this case
            return v[0]
        elif len(v) <= 2:
            # if there are 1 or 2 values, return their mean
            return np.mean(v)
        elif len(v) == 3:
            return mode_of_three(v)
        else:
            v = iteration(v)


def KDE(a, weights):
    xmin = np.min(a)
    xmax = np.max(a)
    resolution = 5000
    dx = (xmax - xmin) / resolution
    xmin -= dx
    xmax += dx
    x = np.linspace(xmin, xmax, resolution + 2)
    estimator = FFTKDE(kernel="gaussian", bw=chen_rule(a, weights))
    y = estimator.fit(a, weights).evaluate(x)
    kde_mode = x[np.argmax(y)]
    return kde_mode


def best_mode(a, weights):
    modes = []
    modes.append(HRM(a))
    modes.append(HSM(a))
    modes.append(KDE(a, weights))
    return mode_of_three(modes)


if __name__ == "__main__":
    model = "FSRS-rs"
    with open(f"./result/{model}.jsonl", "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    weights_list = []
    sizes = []
    n_params = len(DEFAULT_PARAMETER)
    for result in data:
        if type(result["parameters"]) == dict:
            for partition in result["parameters"]:
                for i in range(n_params):
                    if (
                        abs(result["parameters"][partition][i] - DEFAULT_PARAMETER[i])
                        <= 1e-4
                    ):
                        # remove users who have parameters that are close to the default
                        break
                else:
                    weights_list.append(result["parameters"][partition])
                    sizes.append(result["size"])
        else:
            for i in range(n_params):
                if abs(result["parameters"][i] - DEFAULT_PARAMETER[i]) <= 1e-4:
                    # remove users who have parameters that are close to the default
                    break
            else:
                weights_list.append(result["parameters"])
                sizes.append(result["size"])

    weights = np.array(weights_list)
    # sizes = np.sqrt(np.array(sizes))
    print(weights.shape)
    pathlib.Path("./plots").mkdir(parents=True, exist_ok=True)
    for i in range(n_params):
        plt.hist(weights[:, i], bins=128, log=False)
        median = np.median(weights[:, i])
        mean = np.mean(weights[:, i])
        # mode = best_mode(weights[:, i], sizes)
        mode = best_mode(weights[:, i], np.ones_like(weights[:, i]))
        plt.ylim(ymin=0.1)
        plt.axvline(
            median,
            color="orange",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.2f}",
        )
        plt.axvline(
            mean,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean:.2f}",
        )
        plt.axvline(
            mode,
            color="purple",
            linestyle="dashed",
            linewidth=2,
            label=f"Mode: {mode:.2f}",
        )
        plt.xlabel("Parameter value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(f"w[{i}]")
        plt.savefig(f"./plots/w[{i}].png")
        plt.clf()
