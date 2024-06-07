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

- Log Loss (also known as Binary Cross Entropy): Utilized primarily for its applicability in binary classification problems, log loss serves as a measure of the discrepancies between predicted probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm approximates the true recall probabilities, making it an important metric for model evaluation in spaced repetition systems. Log Loss ranges from 0 to infinity, lower is better. 
- Root Mean Square Error in Bins (RMSE (bins)): This is a metric engineered for the SRS benchmark. In this approach, predictions and review outcomes are grouped into bins based on three features: the interval length, the number of reviews, and the number of lapses. Within each bin, the squared difference between the average predicted probability of recall and the average recall rate is calculated. These values are then weighted according to the sample size in each bin, and then the final weighted root mean square error is calculated. This metric provides a nuanced understanding of model performance across different probability ranges. For more details, you can read [The Metric](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Metric). RMSE (bins) ranges from 0 to 1, lower is better.

If you are unsure what metric to look at, look at RMSE (bins). Its value can be interpreted as "the average difference between the predicted probability of recalling a card and the measured probability". For example, if RMSE (bins)=0.05, it means that that algorithm is, on average, wrong by 5% when predicting the probability of recall.

### Models

- FSRS v3: the first version of the FSRS algorithm that people actually used.
- FSRS v4: the upgraded version of FSRS, made better with help from the community.
- FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed. This benchmark also includes FSRS-4.5 with default parameters (which have been obtained by running FSRS-4.5 on all 20 thousand collections) and FSRS-4.5 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
- FSRS-rs: the Rust port of FSRS-4.5. See also: https://github.com/open-spaced-repetition/fsrs-rs
- GRU: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
    - GRU-P: a variant of GRU that removes forgetting curve and predicts the probability of recall directly.
- DASH: the model proposed in [here](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/zp38wc97m). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this model: DASH[MCM] and DASH[ACT-R]. For more details, you can read the paper [here](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
- ACT-R: the model proposed in [here](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression. For more details, you can read the paper [here](https://github.com/duolingo/halflife-regression).
- Transformer: a type of neural network that has gained popularity in recent years due to its superior performance in natural language processing. ChatGPT is based on this architecture.
- SM-2: one of the early algorithms used by SuperMemo, the first spaced repetition software. It was developed more than 30 years ago, and it's still popular today. [Anki's default algorithm is based on SM-2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html), [Mnemosyne](https://mnemosyne-proj.org/principles.php) also uses it.
- NN-17: a neural network approximation of [SM-17](https://supermemo.guru/wiki/Algorithm_SM-17). It has a comparable number of parameters, and according to our estimates, it performs similarly to SM-17.
- AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline.

For more details about the FSRS algorithm, read this wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of users: 19,990.

Total number of reviews for evaluation: 707,964,360. Same-day reviews are excluded; only one review per day (the first chronologically) is used by each algorithm. This is why the number of reviews is significantly lower than the 1.5 billion mentioned earlier. Plus, an outlier filter is also used.

The following tables show the weighted means and the 99% confidence intervals. The best result is highlighted in **bold**. The rightmost column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included.

### Weighted by the number of reviews

| Model | Parameters | Log Loss | RMSE (bins) |
| --- | --- | --- | --- |
| **GRU-P** | 297 | **0.32±0.005** | **0.045±0.0009** |
| FSRS-4.5 | 17 | 0.33±0.005 | 0.053±0.0010 |
| FSRS-rs | 17 | 0.33±0.006 | 0.055±0.0013 |
| FSRS v4 | 17 | 0.33±0.006 | 0.058±0.0013 |
| DASH | 9 | 0.33±0.005 | 0.063±0.0011 |
| DASH[MCM] | 9 | 0.33±0.005 | 0.064±0.0011 |
| DASH[ACT-R] | 5 | 0.34±0.006 | 0.068±0.0014 |
| FSRS v3 | 13 | 0.36±0.007 | 0.072±0.0015 |
| FSRS-4.5 (only pretrain) | 4 | 0.34±0.006 | 0.075±0.0018 |
| GRU | 39 | 0.38±0.007 | 0.080±0.0017 |
| NN-17 | 39 | 0.35±0.007 | 0.081±0.0016 |
| FSRS-4.5 (default parameters) | 0 | 0.35±0.006 | 0.086±0.0021 |
| ACT-R | 5 | 0.36±0.006 | 0.092±0.0022 |
| AVG | 0 | 0.36±0.006 | 0.093±0.0023 |
| HLR | 3 | 0.41±0.009 | 0.107±0.0021 |
| SM-2 | 0 | 0.54±0.013 | 0.149±0.0032 |
| Transformer | 127 | 0.52±0.012 | 0.187±0.0036 |

### Unweighted

| Model | Parameters | Log Loss | RMSE (bins) |
| --- | --- | --- | --- |
| **GRU-P** | 297 | **0.345±0.0030** | **0.068±0.0008** |
| FSRS-4.5 | 17 | 0.352±0.0031 | 0.077±0.0009 |
| FSRS-rs | 17 | 0.353±0.0032 | 0.077±0.0009 |
| FSRS v4 | 17 | 0.362±0.0033 | 0.084±0.0010 |
| DASH | 9 | 0.358±0.0031 | 0.085±0.0009 |
| DASH[MCM] | 9 | 0.358±0.0032 | 0.087±0.0009 |
| DASH[ACT-R] | 5 | 0.362±0.0033 | 0.090±0.0011 |
| FSRS-4.5 (only pretrain) | 4 | 0.363±0.0032 | 0.091±0.0009 |
| FSRS v3 | 13 | 0.422±0.0047 | 0.108±0.0014 |
| FSRS-4.5 (default parameters) | 0 | 0.379±0.0033 | 0.108±0.0012 |
| NN-17 | 39 | 0.380±0.0035 | 0.109±0.0014 |
| AVG | 0 | 0.385±0.0036 | 0.111±0.0012 |
| GRU | 39 | 0.44±0.005 | 0.111±0.0013 |
| ACT-R | 5 | 0.396±0.0041 | 0.116±0.0013 |
| HLR | 3 | 0.46±0.005 | 0.129±0.0014 |
| Transformer | 127 | 0.55±0.007 | 0.192±0.0018 |
| SM-2 | 0 | 0.71±0.013 | 0.202±0.0022 |

Averages weighted by the number of reviews are more representative of "best case" performance when plenty of data is available. Since all algorithms perform better when there's a lot of data to learn from, weighting by n(reviews) biases the average towards lower values.

Unweighted averages are more representative of "average case" performance. In reality, not every user will have hundreds of thousands of reviews, so the algorithm won't always be able to reach its full potential.

The image below shows the p-values obtained by running the Wilcoxon signed-rank test on the RMSE of all pairs of algorithms. Red means that the row algorithm performs worse than the corresponding column algorithm, and green means that the row algorithm performs better than the corresponding column algorithm. Grey means that the p-value is >0.01, and we cannot conclude that one algorithm performs better than the other.

Almost all p-values are extremely small, many orders of magnitude smaller than 0.01. Of course, p-values this low beg the question, "Can we even trust these values?". `scipy.stats.wilcoxon` itself uses an approximation for n>50, and our modified implementation uses an approximation to return the decimal logarithm of the p-value rather than the p-value itself, to avoid the limitations of 64-bit floating point numbers. So it's an approximation of an approximation. But more importantly, this test is not weighted, meaning that it doesn't take into account the fact that RMSE depends on the number of reviews.

Overall, these p-values can be trusted on a qualitative (but not quantitative) level.

![Wilcoxon, 19990 collections](./plots/Wilcoxon-19990-collections.png)

## Default Parameters

FSRS-4.5:

```
0.4872, 1.4003, 3.7145, 13.8206,
5.1618, 1.2298, 0.8975, 0.031,
1.6474, 0.1367, 1.0461,
2.1072, 0.0793, 0.3246, 1.587,
0.2272, 2.8755
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
