# SRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it spreads out your study sessions over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something, so they can schedule a review just in time.

This benchmark is a tool to test how well different algorithms do at predicting your memory. It compares several algorithms to see which ones give the most accurate predictions.

## Dataset

The dataset for the SRS benchmark comes from 20 thousand people who use Anki, a flashcard app. In total, this dataset contains information about ~1.5 billion reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/FSRS-Anki-20k](https://huggingface.co/datasets/open-spaced-repetition/FSRS-Anki-20k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by giving the algorithm future information it shouldn't have. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

Note: TimeSeriesSplit will remove the first split from evaluation. This is because the first split is used for training, and we don't want to evaluate the algorithm on the same data it was trained on.

### Metrics

We use two metrics in the SRS benchmark to evaluate how well these algorithms work: log loss and a custom RMSE that we call RMSE (bins).

- Log Loss (also known as Binary Cross Entropy): Utilized primarily for its applicability in binary classification problems, log loss serves as a measure of the discrepancies between predicted probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm approximates the true recall probabilities, making it an important metric for model evaluation in spaced repetition systems.
- Root Mean Square Error in Bins (RMSE (bins)): This is a metric engineered for the SRS benchmark. In this approach, predictions and review outcomes are grouped into bins based on three features: the interval length, the number of reviews, and the number of lapses. Within each bin, the squared difference between the average predicted probability of recall and the average recall rate is calculated. These values are then weighted according to the sample size in each bin, and then the final weighted root mean square error is calculated. This metric provides a nuanced understanding of model performance across different probability ranges. For more details, you can read [The Metric](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Metric).

Smaller is better. If you are unsure what metric to look at, look at RMSE (bins). That value can be interpreted as "the average difference between the predicted probability of recalling a card and the measured probability". For example, if RMSE (bins)=0.05, it means that that algorithm is, on average, wrong by 5% when predicting the probability of recall.

### Models

- FSRS v3: the first version of the FSRS algorithm that people actually used.
- FSRS v4: the upgraded version of FSRS, made better with help from the community.
- FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed. This benchmark also includes FSRS-4.5 with default parameters (which have been obtained by running FSRS-4.5 on all 20 thousand collections) and FSRS-4.5 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
- FSRS-rs: the Rust port of FSRS-4.5. See also: https://github.com/open-spaced-repetition/fsrs-rs
- GRU: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
- DASH: the model proposed in [here](https://doi.org/10.4324/9781315413570-8). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this model: DASH[MCM] and DASH[ACT-R]. For more details, you can read the paper [here](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
- ACT-R: the model proposed in [here](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression. For more details, you can read the paper [here](https://github.com/duolingo/halflife-regression).
- SM-2: one of the early algorithms used by SuperMemo, the first spaced repetition software. It was developed more than 30 years ago, and it's still popular today. [Anki's default algorithm is based on SM-2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html), [Mnemosyne](https://mnemosyne-proj.org/principles.php) also uses it.

For more details about the FSRS algorithm, read this wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of users: 19,990.

Total number of reviews for evaluation: 708,151,820. Same-day reviews are excluded; only one review per day (the first chronologically) is used by each algorithm. This is why the number of reviews is significantly lower than the 1.5 billion mentioned earlier. Plus, an outlier filter is also used.

The following tables show the weighted means and the 99% confidence intervals. The best result is highlighted in **bold**. The rightmost column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included.

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE (bins) | Parameters |
| --- | --- | --- | --- |
| **FSRS-4.5** | **0.32±0.005** | **0.052±0.0010** | 17 |
| FSRS-rs | 0.32±0.005 | 0.053±0.0013 | 17 |
| FSRS v4 | 0.33±0.006 | 0.057±0.0014 | 17 |
| DASH | 0.33±0.005 | 0.061±0.0011 | 9 |
| DASH[MCM] | 0.33±0.005 | 0.063±0.0011 | 9 |
| DASH[ACT-R] | 0.33±0.006 | 0.066±0.0014 | 5 |
| FSRS v3 | 0.36±0.007 | 0.070±0.0015 | 13 |
| FSRS-4.5 (only pretrain) | 0.34±0.006 | 0.076±0.0018 | 4 |
| GRU | 0.38±0.007 | 0.080±0.0016 | 36 |
| FSRS-4.5 (default parameters) | 0.35±0.006 | 0.087±0.0020 | 0 |
| ACT-R | 0.35±0.006 | 0.090±0.0022 | 5 |
| HLR | 0.40±0.008 | 0.105±0.0020 | 3 |
| SM-2 | 0.54±0.013 | 0.149±0.0031 | 0 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE (bins) | Parameters |
| --- | --- | --- | --- |
| **FSRS-4.5** | **0.346±0.0031** | **0.072±0.0008** | 17 |
| FSRS-rs | 0.347±0.0030 | 0.073±0.0008 | 17 |
| FSRS v4 | 0.354±0.0031 | 0.078±0.0009 | 17 |
| DASH | 0.350±0.0030 | 0.080±0.0009 | 9 |
| DASH[MCM] | 0.351±0.0029 | 0.082±0.0009 | 9 |
| DASH[ACT-R] | 0.354±0.0031 | 0.085±0.0010 | 5 |
| FSRS-4.5 (only pretrain) | 0.360±0.0032 | 0.089±0.0009 | 4 |
| FSRS v3 | 0.404±0.0041 | 0.101±0.0013 | 13 |
| FSRS-4.5 (default parameters) | 0.375±0.0033 | 0.106±0.0011 | 0 |
| GRU | 0.431±0.0049 | 0.107±0.0012 | 36 |
| ACT-R | 0.381±0.0035 | 0.109±0.0012 | 5 |
| HLR | 0.439±0.0047 | 0.123±0.0013 | 3 |
| SM-2 | 0.68±0.011 | 0.195±0.0020 | 0 |

Averages weighted by the number of reviews are more representative of "best case" performance when plenty of data is available. Since all algorithms perform better when there's a lot of data to learn from, weighting by n(reviews) biases the average towards lower values.
Averages weighted by the natural logarithm of the number of reviews are more representative of "average case" performance. In reality, not every user will have hundreds of thousands of reviews, so the algorithm won't always be able to reach its full potential.

The image below shows the p-values obtained by running the Wilcoxon signed-rank test on the RMSE of all pairs of algorithms. Red means that the row algorithm performs worse than the corresponding column algorithm, and green means that the row algorithm performs better than the corresponding column algorithm. The top row is green because FSRS-4.5 outperforms all other algorithms; the bottom row is red because SM-2 performs worse than any other algorithm. Grey means that the p-value is >0.01, and we cannot conclude that one algorithm performs better than the other.

Almost all p-values are extremely small, many orders of magnitude smaller than 0.01. Of course, p-values this low beg the question, "Can we even trust these values?". `scipy.stats.wilcoxon` itself uses an approximation for n>50, and our modified implementation uses an approximation to return the decimal logarithm of the p-value rather than the p-value itself, to avoid the limitations of 64-bit floating point numbers. So it's an approximation of an approximation. But more importantly, this test is not weighted, meaning that it doesn't take into account the fact that RMSE depends on the number of reviews.
Overall, these p-values can be trusted on a qualitative (but not quantitative) level.

![Wilcoxon, 19990 collections](./plots/Wilcoxon-19990-collections.png)

## Median Parameters

FSRS-4.5:

```
0.5701, 1.448, 4.1396, 11.0112,
5.1674, 1.2378, 0.8798, 0.0362,
1.629, 0.1342, 1.0309,
2.099, 0.0839, 0.3204, 1.5704,
0.219, 2.8538
```

FSRS-rs:

```
0.5702, 1.4484, 4.1396, 11.01,
5.1527, 1.2386, 0.8673, 0.0615,
1.6406, 0.1319, 1.0178,
2.1669, 0.0673, 0.3318, 1.5551,
0.2218, 2.9248
```

## Comparisons with SuperMemo 15/16/17

Please go to:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)


## Run the benchmark

### Requirements

Dataset (tiny): https://github.com/open-spaced-repetition/fsrs-benchmark/issues/28#issuecomment-1876196288

Dependencies:

```bash
pip install -r requirements.txt
```

### Commands

FSRS-4.5:

```bash
python script.py
```

FSRS-4.5 with default parameters:

```bash
DRY_RUN=1 python script.py
```

FSRS-rs:

```bash
FSRS_RS=1 FSRS_NO_OUTLIER=1 PYTHONPATH=~/Codes/anki/out/pylib:~/Codes/anki/pylib python script.py
```

> Please change the `PYTHONPATH` variable to the path of your Anki source code.

FSRSv4/FSRSv3/HLR/LSTM/SM2:

```bash
MODEL=FSRSv4 python other.py
```

> Please change the `MODEL` variable to `FSRSv3`, `HLR`, `GRU`, or `SM2` to run the corresponding model.

Dev model in fsrs-optimizer:

```bash
DEV_MODE=1 python script.py
```

> Please place the fsrs-optimizer repository in the same directory as this repository.
