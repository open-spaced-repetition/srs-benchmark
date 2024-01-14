# FSRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it spreads out your study sessions over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something so they can schedule a review just in time.

FSRS benchmark is a tool to test how well different algorithms do at predicting your memory. It compares several algorithms to see which ones give the most accurate predictions.

## Dataset

The dataset for the FSRS benchmark comes from 20k people who use Anki, a flashcard app. In total, there are 1.5B times people reviewed flashcards.

## Evaluation

### Data Split

In the FSRS benchmark, we use a tool called TimeSeriesSplit. This is part of the sklearn library used for machine learning. The tool helps us split the data by time—older stuff is used for training, and newer stuff for testing. That way, we don't accidentally cheat by giving the algorithm future info it shouldn't have. In practice, we use past study sessions to predict future ones. This makes TimeSeriesSplit a good fit for our benchmark.

Note: TimeSeriesSplit will remove the first split from evaluation. This is because the first split is used for training, and we don't want to evaluate the algorithm on the same data it was trained on.

### Metrics

We use three metrics in the FSRS benchmark to evaluate how well these algorithms work: Log Loss, RMSE, and a custom RMSE that we call RMSE(bins).

- Logarithmic Loss (Log Loss): Utilized primarily for its applicability in binary classification problems, log_loss serves as a measure of the discrepancies between predicted probabilities of recall and actual recall events. It quantifies how well the algorithm approximates the true recall probabilities, making it a critical metric for model evaluation in spaced repetition systems.
- Root Mean Square Error (RMSE): Adopted from established metric in the SuperMemo, RMSE provides a holistic measure of model prediction errors. The metric assesses the average magnitude of the differences between predicted and actual recall probabilities, thereby indicating the algorithm's reliability in general terms.
- Weighted Root Mean Square Error in Bins (RMSE(bins)): This is a bespoke metric engineered for the FSRS benchmark. In this approach, predictions and actual recall events are grouped into bins according to the predicted probabilities of recall. Within each bin, the RMSE between the average predicted probability and the average actual recall rate is calculated. These RMSE values are then weighted according to the sample size in each bin, providing a nuanced understanding of model performance across different probability ranges.

Smaller is better. If you are unsure what number to look at, look at RMSE (bins). That value can be interpreted as "the average difference between the predicted probability of recalling a card and the measured probability". For example, if RMSE (bins)=0.05, it means that that algorithm is, on average, wrong by 5% when predicting the probability of recall.

### Models

- FSRS v3: the first version of the FSRS algorithm that people actually used.
- FSRS v4: the upgraded version of FSRS, made better with help from the community.
- FSRS-4.5: the minorly improved version based on FSRS v4. The shape of forgetting curve is changed.
- FSRS rs: the Rust port of FSRS v4, it's simplified due to the limitations of the Rust-based deep learning framework. See also: https://github.com/open-spaced-repetition/fsrs-rs
- LSTM: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks. Our implementation includes 489 parameters.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression, for more details, you can check out the paper [here](https://github.com/duolingo/halflife-regression).
- SM2: the algorithm used by SuperMemo, the first spaced repetition software. It's a classic in the field of spaced repetition, and it's still popular today. [Anki's default algorithm is based on SM2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html).

For all the nerdy details about FSRS, there's a wiki page you can check: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)

## Result

Total number of users: 19854

Total number of reviews for evaluation: 697,851,710

The following tables represent the weighted means and the 99% confidence intervals. The best result is highlighted in **bold**.

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| **FSRS-4.5** | **0.33±0.006** | **0.299±0.0035** | **0.044±0.0011** |
| FSRS rs | 0.33±0.006 | 0.301±0.0035 | 0.049±0.0015 |
| FSRS v4 | 0.34±0.006 | 0.301±0.0036 | 0.053±0.0015 |
| FSRS-4.5 (default parameters) | 0.36±0.006 | 0.310±0.0036 | 0.076±0.0021 |
| LSTM | 0.39±0.007 | 0.312±0.0037 | 0.079±0.0019 |
| FSRS v3 | 0.41±0.008 | 0.321±0.0039 | 0.100±0.0026 |
| SM2 | 0.56±0.014 | 0.353±0.0040 | 0.167±0.0032 |
| HLR | 0.77±0.020 | 0.377±0.0046 | 0.205±0.0048 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| **FSRS-4.5** | **0.358±0.0031** | **0.314±0.0019** | **0.065±0.0008** |
| FSRS rs | 0.362±0.0030 | 0.316±0.0019 | 0.069±0.0009 |
| FSRS v4 | 0.368±0.0033 | 0.317±0.0019 | 0.076±0.0010 |
| FSRS-4.5 (default parameters) | 0.386±0.0034 | 0.326±0.0019 | 0.095±0.0011 |
| FSRS v3 | 0.49±0.006 | 0.343±0.0020 | 0.129±0.0015 |
| LSTM | 0.55±0.009 | 0.357±0.0022 | 0.147±0.0020 |
| SM2 | 0.73±0.013 | 0.382±0.0022 | 0.201±0.0019 |
| HLR | 2.26±0.047 | 0.537±0.0032 | 0.404±0.0040 |

## Median Parameters

FSRS-4.5:

```
0.5615, 1.2723, 3.6032, 8.0462,
5.1368, 1.2205, 0.8679, 0.0387,
1.6499, 0.138, 1.0294,
2.1106, 0.0876, 0.3248, 1.4093,
0.219, 2.8403
```

FSRS rs:

```
0.5141, 1.1631, 3.5152, 7.8989,
5.0981, 1.0942, 0.8011, 0.068,
1.6409, 0.1405, 0.9801,
2.1824, 0.0712, 0.3157, 1.3886,
0.2448, 2.8316
```

## Comparisons with SuperMemo 15/16/17

Please go to:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)
