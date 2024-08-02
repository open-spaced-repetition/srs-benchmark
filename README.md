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
- FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed. This benchmark also includes FSRS-4.5 with default parameters (which have been obtained by running FSRS-4.5 on all 20 thousand collections) and FSRS-4.5 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
- FSRS-5: the latest version of FSRS. Unlike the previous versions, it takes into account same-day reviews. Same-day reviews are used only for training, and not for evaluation.
- FSRS-rs: the Rust port of FSRS-5. See also: https://github.com/open-spaced-repetition/fsrs-rs
- GRU: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
    - GRU-P: a variant of GRU that removes forgetting curve and predicts the probability of recall directly.
    - GRU-P-short: same as above, but it also takes into account same-day reviews.
- DASH: the model proposed in [this paper](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/zp38wc97m). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this model: DASH[MCM] and DASH[ACT-R]. For further information, please refer to [this paper](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
- ACT-R: the model proposed in [this paper](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression. For further information, please refer to the [this paper](https://github.com/duolingo/halflife-regression).
- Transformer: a type of neural network that has gained popularity in recent years due to its superior performance in natural language processing. ChatGPT is based on this architecture.
- SM-2: one of the early algorithms used by SuperMemo, the first spaced repetition software. It was developed more than 30 years ago, and it's still popular today. [Anki's default algorithm is based on SM-2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html), [Mnemosyne](https://mnemosyne-proj.org/principles.php) also uses it. This algorithm does not predict the probability of recall natively; therefore, for the sake of the benchmark, the output was modified based on some assumptions about the forgetting curve.
- NN-17: a neural network approximation of [SM-17](https://supermemo.guru/wiki/Algorithm_SM-17). It has a comparable number of parameters, and according to our estimates, it performs similarly to SM-17.
- AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline.

For further information regarding the FSRS algorithm, please refer to the following wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of users: 19,990.

Total number of reviews for evaluation: 707,964,360.
Same-day reviews are excluded except in FSRS-5 and GRU-P-short, i.e., each algorithm uses only one review per day (the first, chronologically). Some reviews are filtered out, for example, the revlog entries created on changing the due date manually or reviewing the cards in a filtered deck with "Reschedule cards based on my answers in this deck" disabled. Finally, an outlier filter is applied. These are the reasons why the number of reviews used for evaluation is significantly lower than the figure of 1.7 billion mentioned earlier. 

The following tables present the means and the 99% confidence intervals. The best result is highlighted in **bold**. The rightmost column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included.

### Weighted by the number of reviews

| Model | Parameters | Log Loss | RMSE (bins) | AUC |
| --- | --- | --- | --- | --- |
| GRU-P-short | 297 | **0.314±0.0052** | **0.0420±0.00086** | **0.707±0.0029** |
| GRU-P | 297 | 0.319±0.0052 | 0.0434±0.00085 | 0.698±0.0029 |
| FSRS-5 | 19 | 0.321±0.0055 | 0.050±0.0012 | 0.700±0.0028 |
| FSRS-4.5 | 17 | 0.325±0.0055 | 0.052±0.0010 | 0.693±0.0028 |
| FSRS-rs | 19 | 0.324±0.0054 | 0.053±0.0015 | 0.692±0.0029 |
| FSRS v4 | 17 | 0.331±0.0056 | 0.058±0.0014 | 0.690±0.0028 |
| DASH | 9 | 0.332±0.0054 | 0.063±0.0011 | 0.643±0.0031 |
| DASH[MCM] | 9 | 0.332±0.0054 | 0.064±0.0011 | 0.644±0.0032 |
| DASH[ACT-R] | 5 | 0.336±0.0055 | 0.066±0.0014 | 0.632±0.0033 |
| FSRS v3 | 13 | 0.361±0.0068 | 0.072±0.0015 | 0.667±0.0030 |
| NN-17 | 39 | 0.346±0.0063 | 0.076±0.0016 | 0.596±0.0032 |
| FSRS-5 (only pretrain) | 4 | 0.348±0.0067 | 0.077±0.0020 | 0.679±0.0030 |
| GRU | 39 | 0.375±0.0071 | 0.079±0.0016 | 0.659±0.0029 |
| FSRS-5 (default parameters) | 0 | 0.356±0.0070 | 0.085±0.0020 | 0.674±0.0030 |
| ACT-R | 5 | 0.356±0.0060 | 0.085±0.0022 | 0.536±0.0032 |
| AVG | 0 | 0.356±0.0059 | 0.086±0.0022 | 0.508±0.0029 |
| HLR | 3 | 0.409±0.0089 | 0.107±0.0021 | 0.631±0.0034 |
| SM-2 | 0 | 0.54±0.013 | 0.147±0.0031 | 0.599±0.0032 |
| Transformer | 127 | 0.52±0.012 | 0.187±0.0037 | 0.515±0.0044 |

### Unweighted

| Model | Parameters | Log Loss | RMSE (bins) | AUC |
| --- | --- | --- | --- | --- |
| **GRU-P** | 297 | 0.345±0.0030 | **0.0656±0.00083** | 0.679±0.0018 |
| GRU-P-short | 297 | **0.341±0.0030** | 0.0659±0.00085 | 0.687±0.0018 |
| FSRS-5 | 19 | 0.347±0.0031 | 0.0712±0.00084 | **0.697±0.0017** |
| FSRS-rs | 19 | 0.348±0.0031 | 0.0726±0.00088 | 0.693±0.0018 |
| FSRS-4.5 | 17 | 0.352±0.0032 | 0.0743±0.00089 | 0.688±0.0017 |
| FSRS v4 | 17 | 0.362±0.0034 | 0.084±0.0010 | 0.685±0.0017 |
| DASH | 9 | 0.358±0.0031 | 0.0846±0.00096 | 0.632±0.0018 |
| DASH[ACT-R] | 5 | 0.362±0.0033 | 0.086±0.0011 | 0.627±0.0019 |
| DASH[MCM] | 9 | 0.358±0.0031 | 0.0871±0.00098 | 0.636±0.0019 |
| FSRS-5 (only pretrain) | 4 | 0.365±0.0034 | 0.0884±0.00095 | 0.689±0.0016 |
| NN-17 | 39 | 0.380±0.0035 | 0.100±0.0013 | 0.570±0.0018 |
| AVG | 0 | 0.385±0.0036 | 0.101±0.0011 | 0.500±0.0018 |
| FSRS-5 (default parameters) | 0 | 0.377±0.0034 | 0.102±0.0011 | 0.688±0.0016 |
| ACT-R | 5 | 0.396±0.0040 | 0.106±0.0012 | 0.525±0.0017 |
| FSRS v3 | 13 | 0.422±0.0045 | 0.108±0.0014 | 0.661±0.0017 |
| GRU | 39 | 0.440±0.0053 | 0.108±0.0013 | 0.650±0.0017 |
| HLR | 3 | 0.456±0.0051 | 0.129±0.0014 | 0.636±0.0019 |
| Transformer | 127 | 0.555±0.0067 | 0.192±0.0019 | 0.527±0.0021 |
| SM-2 | 0 | 0.71±0.012 | 0.199±0.0021 | 0.604±0.0017 |

Averages weighted by the number of reviews are more representative of "best case" performance when plenty of data is available. Since almost all algorithms perform better when there's a lot of data to learn from, weighting by n(reviews) biases the average towards lower values.

Unweighted averages are more representative of "average case" performance. In reality, not every user will have hundreds of thousands of reviews, so the algorithm won't always be able to reach its full potential.

The image below shows the p-values obtained by running the Wilcoxon signed-rank test on the RMSE of all pairs of algorithms. Red means that the row algorithm performs worse than the corresponding column algorithm, and green means that the row algorithm performs better than the corresponding column algorithm. Grey means that the p-value is >0.01, and we cannot conclude that one algorithm performs better than the other.

Almost all p-values are extremely small, many orders of magnitude smaller than 0.01. Of course, p-values this low beg the question, "Can we even trust these values?". `scipy.stats.wilcoxon` itself uses an approximation for n>50, and our modified implementation uses an approximation to return the decimal logarithm of the p-value rather than the p-value itself, to avoid the limitations of 64-bit floating point numbers. So it's an approximation of an approximation. But more importantly, this test is not weighted, meaning that it doesn't take into account the fact that RMSE depends on the number of reviews.

Overall, these p-values can be trusted on a qualitative (but not quantitative) level.

![Wilcoxon, 19990 collections](./plots/Wilcoxon-19990-collections.png)

## Default Parameters

FSRS-5:

```
0.4197, 1.1869, 3.0412, 15.2441,
7.1434, 0.6477, 1.0007, 0.0674,
1.6597, 0.1712, 1.1178,
2.0225, 0.0904, 0.3025, 2.1214,
0.2498, 2.9466,
0.4891, 0.6468
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
