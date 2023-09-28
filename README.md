# FSRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it spreads out your study sessions over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something so they can schedule a review just in time.

FSRS benchmark is a tool to test how well different algorithms do at predicting your memory. It compares several algorithms to see which ones give the most accurate predictions.

## Dataset

The dataset for the FSRS benchmark comes from 72 people who use Anki, a flashcard app. In total, there are 6,240,084 times people reviewed flashcards. The dataset includes over 6 million review records.

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

### Models

- FSRS v3: This is the first version of the FSRS algorithm that people actually used.
- FSRS v4: This one's an upgraded version of FSRS, made better with help from the community.
- LSTM: This is a type of neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks.
- HLR: This is a model proposed by Duolingo. Its full name is Half-Life Regression, for more details, you can check out the paper [here](https://github.com/duolingo/halflife-regression).
- SM2: This is the algorithm used by SuperMemo, the first spaced repetition software. It's a classic in the field of spaced repetition, and it's still popular today. [Anki's default algorithm is based on SM2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html).

For all the nerdy details about FSRS, there's a wiki page you can check: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)

## Result

Total number of users: 71

Total number of reviews for evaluation: 4,632,965

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3874 | 0.3347 | 0.0459 |
| FSRS rs | 0.3910 | 0.3359 | 0.0492 |
| LSTM | 0.4199 | 0.3425 | 0.0662 |
| FSRS v3 | 0.4890 | 0.3633 | 0.1204 |
| SM2 | 0.7317 | 0.4066 | 0.2079 |
| HLR | 0.7951 | 0.4113 | 0.2094 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3820 | 0.3311 | 0.0547 |
| FSRS rs | 0.3862 | 0.3328 | 0.0591 |
| FSRS v3 | 0.5132 | 0.3670 | 0.1326 |
| LSTM | 0.5788 | 0.3752 | 0.1385 |
| SM2 | 0.8847 | 0.4131 | 0.2185 |
| HLR | 2.5175 | 0.5574 | 0.4141 |

## Weights

FSRS v4:

```
1.9337, 3.1175, 7.7239, 37.6583,
4.7122, 1.3422, 1.1666, 0.0394,
1.6075, 0.1969, 1.0985,
2.209, 0.0761, 0.3596, 1.3072,
0.2932, 3.3027
```

FSRS rs:

```
1.7433, 2.8415, 6.5027, 26.241,
4.8353, 1.169, 1.0242, 0.028,
1.6129, 0.1709, 1.0334,
2.3245, 0.0391, 0.3436, 1.3621,
0.2581, 3.2158
```

## Comparisons with SuperMemo 15/16/17

Please go to:
- [fsrs-vs-sm15](https://github.com/open-spaced-repetition/fsrs-vs-sm15)
- [fsrs-vs-sm17](https://github.com/open-spaced-repetition/fsrs-vs-sm17)
