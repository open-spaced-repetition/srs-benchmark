# FSRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it spreads out your study sessions over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something so they can schedule a review just in time.

FSRS benchmark is a tool to test how well different algorithms do at predicting your memory. It compares several algorithms to see which ones give the most accurate predictions.

## Dataset

The dataset for the FSRS benchmark comes from 71 people who use Anki, a flashcard app. In total, there are 6,239,827 times people reviewed flashcards. You can find the dataset in huggingface datasets: https://huggingface.co/datasets/open-spaced-repetition/fsrs-dataset. To download the dataset, you can run the script `download_data.py`.

The data has been filtered to focus on long-term study patterns. For example, if a person reviewed the same flashcard multiple times in one day, only the first review is kept in the dataset. If you're curious about the nitty-gritty details of how the data was prepared, you can check out the code in the file named `build_dataset.py`.

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

Total number of users: 71

Total number of reviews for evaluation: 4,632,965

> 1,606,862 reviews are only used for training.

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3872 | 0.3346 | 0.0454 |
| FSRS rs | 0.3920 | 0.3361 | 0.0495 |
| LSTM | 0.4199 | 0.3425 | 0.0662 |
| FSRS v3 | 0.4890 | 0.3633 | 0.1204 |
| SM2 | 0.7317 | 0.4066 | 0.2079 |
| HLR | 0.7951 | 0.4113 | 0.2094 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3819 | 0.3311 | 0.0543 |
| FSRS rs | 0.3859 | 0.3326 | 0.0582 |
| FSRS v3 | 0.5132 | 0.3670 | 0.1326 |
| LSTM | 0.5788 | 0.3752 | 0.1385 |
| SM2 | 0.8847 | 0.4131 | 0.2185 |
| HLR | 2.5175 | 0.5574 | 0.4141 |

## Median Weights

FSRS v4:

```
0.3904, 0.9717, 2.3, 11.0667,
4.9804, 1.1958, 0.9834, 0.0074,
1.5884, 0.1617, 1.0601,
2.2097, 0.0476, 0.3442, 1.3127,
0.1989, 2.7928
```

FSRS rs:

```
0.3824, 0.9, 2.1198, 10.973,
4.9515, 1.0301, 0.9165, 0.0297,
1.5906, 0.1575, 1.0132,
2.228, 0.064, 0.3429, 1.3407,
0.2419, 2.8759
```

## Comparisons with SuperMemo 15/16/17

Please go to:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)
