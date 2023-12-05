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

If you are unsure what number to look at, look at RMSE (bins). That value can be interpreted as "the average difference between the predicted probability of recalling a card and the measured probability". For example, if RMSE (bins) = 0.05, it means that that algorithm is, on average, wrong by 5% when predicting the probability of recall.

### Models

- FSRS v3: the first version of the FSRS algorithm that people actually used.
- FSRS v4: the upgraded version of FSRS, made better with help from the community.
- FSRS rs: the Rust port of FSRS v4, it's simplified due to the limitations of the Rust-based deep learning framework. See also: https://github.com/open-spaced-repetition/fsrs-rs
- LSTM: a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
- HLR: the model proposed by Duolingo. Its full name is Half-Life Regression, for more details, you can check out the paper [here](https://github.com/duolingo/halflife-regression).
- SM2: the algorithm used by SuperMemo, the first spaced repetition software. It's a classic in the field of spaced repetition, and it's still popular today. [Anki's default algorithm is based on SM2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html).

For all the nerdy details about FSRS, there's a wiki page you can check: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)

## Result

Total number of users: 19995

Total number of reviews for evaluation: 738,221,150

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3360 | 0.3002 | 0.0590 |
| FSRS rs | 0.3404 | 0.3018 | 0.0628 |
| LSTM | 0.4000 | 0.3123 | 0.0859 |
| FSRS v3 | 0.4286 | 0.3226 | 0.1074 |
| SM2 | 0.6339 | 0.3592 | 0.1799 |
| HLR | 0.8175 | 0.3790 | 0.2094 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3658 | 0.3141 | 0.0820 |
| FSRS rs | 0.3705 | 0.3161 | 0.0860 |
| FSRS v3 | 0.5201 | 0.3471 | 0.1425 |
| LSTM | 0.5645 | 0.3562 | 0.1536 |
| SM2 | 0.8702 | 0.3954 | 0.2240 |
| HLR | 2.3669 | 0.5393 | 0.4096 |

## Median Parameters

FSRS v4:

```
0.6782, 1.6906, 4.4709, 15.0409,
4.9446, 1.061, 0.8755, 0.0463,
1.5593, 0.1358, 0.9936,
2.1665, 0.0678, 0.3402, 1.2838,
0.2579, 2.7197
```

FSRS rs:

```
0.5888, 1.4616, 3.8226, 14.1364,
4.9214, 1.0325, 0.8731, 0.0613,
1.57, 0.1395, 0.988,
2.212, 0.0658, 0.3439, 1.3098,
0.2837, 2.7766
```

## Comparisons with SuperMemo 15/16/17

Please go to:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)
