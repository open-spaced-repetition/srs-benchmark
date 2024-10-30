# SRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it distributes your reviews over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something, so they can schedule a review accordingly.

This benchmark is a tool designed to assess the predictive accuracy of various algorithms. A multitude of algorithms are evaluated to find out which ones provide the most accurate predictions.

## Dataset

The dataset for the SRS benchmark comes from 20 thousand people who use Anki, a flashcard app. In total, this dataset contains information about ~1.7 billion reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/FSRS-Anki-20k](https://huggingface.co/datasets/open-spaced-repetition/FSRS-Anki-20k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by giving the algorithm future information it shouldn't have. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

Note: TimeSeriesSplit will remove the first split from evaluation. This is because the first split is used for training, and we don't want to evaluate the algorithm on the same data it was trained on.

### Metrics

We use three metrics in the SRS benchmark to evaluate how well these algorithms work: log loss, AUC, and a custom RMSE that we call RMSE (bins).

- Log Loss (also known as Binary Cross Entropy): Utilized primarily for its applicability in binary classification problems, log loss serves as a measure of the discrepancies between predicted probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm approximates the true recall probabilities, making it an important metric for algorithm evaluation in spaced repetition systems. Log Loss ranges from 0 to infinity, lower is better.
- AUC (Area under the ROC Curve): AUC represents the degree or measure of separability. It tells how much the algorithm is capable of distinguishing between classes. AUC ranges from 0 to 1, however, in practice it's almost always greater than 0.5; higher is better.
- Root Mean Square Error in Bins (RMSE (bins)): This is a metric designed for use in the SRS benchmark. In this approach, predictions and review outcomes are grouped into bins based on three features: the interval length, the number of reviews, and the number of lapses. Within each bin, the squared difference between the average predicted probability of recall and the average recall rate is calculated. These values are then weighted according to the sample size in each bin, and then the final weighted root mean square error is calculated. This metric provides a nuanced understanding of algorithm performance across different probability ranges. For more details, you can read [The Metric](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Metric). RMSE (bins) ranges from 0 to 1, lower is better.

### Algorithms

- FSRS v3: the first version of the FSRS algorithm that people actually used.
- FSRS v4: the upgraded version of FSRS, made better with help from the community.
- FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed.
- FSRS-5: the latest version of FSRS. Unlike the previous versions, it takes into account same-day reviews. Same-day reviews are used only for training, and not for evaluation.
    - FSRS-5 default param.: FSRS-5 with default parameters (which have been obtained by running FSRS-5 on all 20 thousand collections).
    - FSRS-5 pretrain: FSRS-5 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
    - FSRS-5 binary: FSRS which treats `hard` and `easy` ratings as `good`.
- FSRS-rs: the Rust port of FSRS-5. See also: https://github.com/open-spaced-repetition/fsrs-rs
- GRU: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
    - GRU-P: a variant of GRU that removes the forgetting curve and predicts the probability of recall directly.
    - GRU-P-short: same as above, but it also takes into account same-day reviews.
- DASH: the model proposed in [this paper](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/zp38wc97m). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this model: DASH[MCM] and DASH[ACT-R]. For further information, please refer to [this paper](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
- ACT-R: the model proposed in [this paper](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression. For further information, please refer to the [this paper](https://github.com/duolingo/halflife-regression).
- Transformer: a type of neural network that has gained popularity in recent years due to its superior performance in natural language processing. ChatGPT is based on this architecture. Both GRU and Transformer use the same power forgetting curve as FSRS-4.5 and FSRS-5 to make the comparison more fair.
- SM-2: one of the early algorithms used by SuperMemo, the first spaced repetition software. It was developed more than 30 years ago, and it's still popular today. [Anki's default algorithm is based on SM-2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html), [Mnemosyne](https://mnemosyne-proj.org/principles.php) also uses it. This algorithm does not predict the probability of recall natively; therefore, for the sake of the benchmark, the output was modified based on some assumptions about the forgetting curve.
    - SM-2-short: a modified implementation that also uses same-day reviews.
- NN-17: a neural network approximation of [SM-17](https://supermemo.guru/wiki/Algorithm_SM-17). It has a comparable number of parameters, and according to our estimates, it performs similarly to SM-17.
- Ebisu v2: [an algorithm that uses Bayesian statistics](https://fasiha.github.io/ebisu/) to update its estimate of memory half-life after every review.
- AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline.

For further information regarding the FSRS algorithm, please refer to the following wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of users: 19,990.

Total number of reviews for evaluation: 702,721,850.
Same-day reviews are excluded except in FSRS-5, GRU-P-short and SM-2-short, i.e., each algorithm uses only one review per day (the first, chronologically). Some reviews are filtered out, for example, the revlog entries created by changing the due date manually or reviewing cards in a filtered deck with "Reschedule cards based on my answers in this deck" disabled. Finally, an outlier filter is applied. These are the reasons why the number of reviews used for evaluation is significantly lower than the figure of 1.7 billion mentioned earlier. 

The following tables present the means and the 99% confidence intervals. The best result is highlighted in **bold**. The rightmost column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included.

### Weighted by the number of reviews

| Model | Parameters | LogLoss | RMSE (bins) | AUC |
| --- | --- | --- | --- | --- |
| **GRU-P-short** | 297 | **0.313±0.0051** | **0.0420±0.00085** | **0.707±0.0029** |
| GRU-P | 297 | 0.318±0.0053 | 0.0435±0.00091 | 0.697±0.0027 |
| FSRS-5 | 19 | 0.319±0.0054 | 0.049±0.0010 | 0.702±0.0027 |
| FSRS-rs | 19 | 0.319±0.0052 | 0.049±0.0010 | 0.702±0.0026 |
| FSRS-4.5 | 17 | 0.324±0.0053 | 0.052±0.0011 | 0.693±0.0027 |
| FSRS-5 binary | 15 | 0.325±0.0053 | 0.053±0.0011 | 0.680±0.0031 |
| FSRS v4 | 17 | 0.329±0.0056 | 0.057±0.0012 | 0.690±0.0027 |
| DASH | 9 | 0.331±0.0053 | 0.060±0.0010 | 0.642±0.0031 |
| DASH[MCM] | 9 | 0.331±0.0054 | 0.062±0.0011 | 0.644±0.0030 |
| GRU | 39 | 0.337±0.0056 | 0.062±0.0012 | 0.672±0.0028 |
| DASH-short | 9 | 0.330±0.0052 | 0.063±0.0012 | 0.642±0.0031 |
| DASH[ACT-R] | 5 | 0.334±0.0055 | 0.065±0.0012 | 0.632±0.0031 |
| FSRS-5 pretrain | 4 | 0.335±0.0057 | 0.070±0.0015 | 0.689±0.0028 |
| FSRS v3 | 13 | 0.360±0.0068 | 0.070±0.0015 | 0.667±0.0029 |
| NN-17 | 39 | 0.346±0.0069 | 0.075±0.0015 | 0.595±0.0031 |
| FSRS-5 default param. | 0 | 0.344±0.0060 | 0.078±0.0017 | 0.684±0.0027 |
| ACT-R | 5 | 0.354±0.0057 | 0.084±0.0019 | 0.536±0.0030 |
| AVG | 0 | 0.354±0.0059 | 0.085±0.0019 | 0.508±0.0029 |
| HLR | 3 | 0.404±0.0079 | 0.102±0.0020 | 0.632±0.0034 |
| SM-2-short | 0 | 0.50±0.011 | 0.124±0.0028 | 0.592±0.0032 |
| SM-2 | 0 | 0.54±0.012 | 0.147±0.0029 | 0.599±0.0031 |
| Ebisu-v2 | 0 | 0.445±0.0081 | 0.157±0.0025 | 0.592±0.0033 |
| Transformer | 127 | 0.439±0.0078 | 0.164±0.0031 | 0.516±0.0043 |

### Unweighted

| Model | Parameters | Log Loss | RMSE (bins) | AUC |
| --- | --- | --- | --- | --- |
| GRU-P | 297 | 0.345±0.0030 | **0.0655±0.00082** | 0.679±0.0017 |
| GRU-P-short | 297 | **0.340±0.0030** | 0.0659±0.00084 | 0.687±0.0019 |
| FSRS-5 | 19 | 0.346±0.0031 | 0.0710±0.00085 | **0.698±0.0017** |
| FSRS-rs | 19 | 0.346±0.0031 | 0.0710±0.00084 | 0.698±0.0017 |
| FSRS-4.5 | 17 | 0.352±0.0032 | 0.0742±0.00088 | 0.688±0.0017 |
| FSRS-5 binary | 15 | 0.355±0.0031 | 0.0776±0.00095 | 0.673±0.0019 |
| DASH | 9 | 0.358±0.0031 | 0.0810±0.00095 | 0.632±0.0018 |
| FSRS v4 | 17 | 0.362±0.0033 | 0.082±0.0010 | 0.685±0.0016 |
| DASH-short | 9 | 0.357±0.0031 | 0.0826±0.00095 | 0.626±0.0020 |
| DASH[MCM] | 9 | 0.358±0.0030 | 0.0831±0.00094 | 0.636±0.0019 |
| FSRS-5 pretrain | 4 | 0.358±0.0032 | 0.0853±0.00089 | 0.693±0.0016 |
| DASH[ACT-R] | 5 | 0.362±0.0033 | 0.086±0.0011 | 0.627±0.0019 |
| GRU | 39 | 0.373±0.0033 | 0.089±0.0010 | 0.660±0.0017 |
| FSRS-5 default param. | 0 | 0.371±0.0033 | 0.100±0.0011 | 0.692±0.0016 |
| NN-17 | 39 | 0.380±0.0035 | 0.100±0.0013 | 0.570±0.0018 |
| AVG | 0 | 0.385±0.0036 | 0.101±0.0011 | 0.500±0.0018 |
| ACT-R | 5 | 0.395±0.0040 | 0.106±0.0012 | 0.524±0.0018 |
| FSRS v3 | 13 | 0.422±0.0046 | 0.106±0.0013 | 0.661±0.0017 |
| HLR | 3 | 0.456±0.0051 | 0.124±0.0013 | 0.636±0.0018 |
| Ebisu-v2 | 0 | 0.482±0.0053 | 0.159±0.0015 | 0.602±0.0018 |
| Transformer | 127 | 0.468±0.0047 | 0.164±0.0016 | 0.527±0.0021 |
| SM-2-short | 0 | 0.63±0.011 | 0.166±0.0019 | 0.594±0.0019 |
| SM-2 | 0 | 0.71±0.013 | 0.199±0.0021 | 0.604±0.0018 |

Averages weighted by the number of reviews are more representative of "best case" performance when plenty of data is available. Since almost all algorithms perform better when there's a lot of data to learn from, weighting by n(reviews) biases the average towards lower values.

Unweighted averages are more representative of "average case" performance. In reality, not every user will have hundreds of thousands of reviews, so the algorithm won't always be able to reach its full potential.

### Superiority

The metrics presented above can be difficult to interpret. In order to make it easier to understand how algorithms perform relative to each other, the image below shows the percentage of users for whom algorithm A (row) has a lower RMSE than algorithm B (column). For example, GRU-P-short has a 94.5% superiority over the Transformer, meaning that for 94.5% of all collections in this benchmark, GRU-P-short can estimate the probability of recall more accurately than the Transformer. This is based on 19,990 collections.

![Superiority, 19990](./plots/Superiority,%2019990.png)

You may have noticed that FSRS-5 has a 99.0% superiority over SM-2, meaning that for 99.0% of users, RMSE will be lower with FSRS-5 than with SM-2. But please remember that SM-2 wasn’t designed to predict probabilities, and the only reason it does that in this benchmark is because extra formulas were added on top of it.

### Statistical significance

The image below shows the p-values obtained by running the Wilcoxon signed-rank test on the RMSE of all pairs of algorithms. Red means that the row algorithm performs worse than the corresponding column algorithm, and green means that the row algorithm performs better than the corresponding column algorithm. Grey means that the p-value is >0.01, and we cannot conclude that one algorithm performs better than the other.

Almost all p-values are extremely small, many orders of magnitude smaller than 0.01. Of course, p-values this low beg the question, "Can we even trust these values?". `scipy.stats.wilcoxon` itself uses an approximation for n>50, and our modified implementation uses an approximation to return the decimal logarithm of the p-value rather than the p-value itself, to avoid the limitations of 64-bit floating point numbers. So it's an approximation of an approximation. But more importantly, this test is not weighted, meaning that it doesn't take into account the fact that RMSE depends on the number of reviews.

Overall, these p-values can be trusted on a qualitative (but not quantitative) level.

![Wilcoxon, 19990 collections](./plots/Wilcoxon-19990-collections.png)

## Default Parameters

FSRS-5:

```
0.40255, 1.18385, 3.173, 15.69105,
7.1949, 0.5345, 1.4604, 0.0046,
1.54575, 0.1192, 1.01925,
1.9395, 0.11, 0.29605, 2.2698,
0.2315, 2.9898,
0.51655, 0.6621
```

## Comparisons with SuperMemo 15/16/17

Please refer to the following repositories:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)


## How to run the benchmark

### Requirements

Dataset (tiny): https://github.com/open-spaced-repetition/fsrs-benchmark/issues/28#issuecomment-1876196288

Dependencies:

```bash
pip install -r requirements.txt
```

### Commands

FSRS-5:

```bash
python script.py
```

FSRS-5 with default parameters:

```bash
DRY_RUN=1 python script.py
```

FSRS-5 with only the first 4 parameters optimized:

```bash
PRETRAIN=1 python script.py
```

FSRS-rs:

```bash
FSRS_RS=1 FSRS_NO_OUTLIER=1 PYTHONPATH=~/Codes/anki/out/pylib:~/Codes/anki/pylib python script.py
```

> Please change the `PYTHONPATH` variable to the path of your Anki source code.

Dev model in fsrs-optimizer:

```bash
DEV_MODE=1 python script.py
```

> Please place the fsrs-optimizer repository in the same directory as this repository.

Set the number of threads:

```bash
THREADS=4 python script.py
```

Save the raw predictions:

```bash
RAW=1 python script.py
```

Save the detailed results:

```bash
FILE=1 python script.py
```

Save the analyzing charts:

```bash
PLOT=1 python script.py
```

Benchmark FSRSv4/FSRSv3/HLR/LSTM/SM2:

```bash
MODEL=FSRSv4 python other.py
```

> Please change the `MODEL` variable to `FSRSv3`, `HLR`, `GRU`, or `SM2` to run the corresponding model.
