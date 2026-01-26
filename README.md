# SRS Benchmark

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of cramming all at once, it distributes your reviews over time. To make this efficient, these algorithms try to understand how your memory works. They aim to predict when you're likely to forget something, so they can schedule a review accordingly.

This benchmark is designed to assess the predictive accuracy of various algorithms. A multitude of algorithms are evaluated to find out which ones provide the most accurate predictions.

**We will evaluate your algorithm! [Open a GitHub issue](https://github.com/open-spaced-repetition/srs-benchmark/issues/new) or contact [L-M-Sherlock](https://github.com/L-M-Sherlock).**

## Dataset

~~The dataset for the SRS benchmark comes from 20 thousand people who use Anki, a flashcard app. In total, this dataset contains information about \~1.7 billion reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/FSRS-Anki-20k](https://huggingface.co/datasets/open-spaced-repetition/FSRS-Anki-20k).~~

The dataset for the SRS benchmark comes from 10 thousand users who use Anki, a flashcard app. In total, this dataset contains information about ~727 million reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by giving the algorithm future information it shouldn't have. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

Note: TimeSeriesSplit will remove the first split from evaluation. This is because the first split is used for training, and we don't want to evaluate the algorithm on the same data it was trained on.

RWKV and RMSE-BINS-EXPLOIT do not use TimeSeriesSplit.

### Metrics

We use three metrics in the SRS benchmark to evaluate how well these algorithms work: Log Loss, AUC, and a custom RMSE that we call RMSE (bins).

- Log Loss (also known as Binary Cross Entropy): used primarily in binary classification problems, Log Loss serves as a measure of the discrepancies between predicted probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm approximates the true recall probabilities. Log Loss ranges from 0 to infinity, lower is better.
- Root Mean Square Error in Bins (RMSE (bins)): this is a metric designed for use in the SRS benchmark. In this approach, predictions and review outcomes are grouped into bins based on three features: the interval length, the number of reviews, and the number of lapses. Within each bin, the squared difference between the average predicted probability of recall and the average recall rate is calculated. These values are then weighted according to the sample size in each bin, and then the final weighted root mean square error is calculated. This metric provides a nuanced understanding of algorithm performance across different probability ranges. For more details, you can read [The Metric](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Metric). RMSE (bins) ranges from 0 to 1, lower is better.
- AUC (Area under the ROC Curve): this metric tells us how much the algorithm is capable of distinguishing between classes, which in this case are "successful recall" and "memory lapse". AUC ranges from 0 to 1, however, in practice it's almost always greater than 0.5; higher is better.

Log Loss and RMSE (bins) measure calibration: how well predicted probabilities of recall match the real data. AUC measures discrimination: how well the algorithm can tell two (or more, generally speaking) classes apart. AUC can be good (high) even if Log Loss and RMSE are poor.

### Algorithms and algorithm families

- Two component or three component* model of memory:
    - FSRS v1 and v2: the initial experimental versions of FSRS.
    - FSRS v3: the first official release of the FSRS algorithm, made available as a custom scheduling script.
    - FSRS v4: the upgraded version of FSRS, made better with help from the community.
    - FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed.
    - FSRS-5: the upgraded version of FSRS. Unlike the previous versions, it uses the same-day review data. Same-day reviews are used only for training, and not for evaluation.
    - FSRS-6: the latest version of FSRS. The formula for handling same-day reviews has been improved. More importantly, FSRS-6 has an optimizable parameter that controls the flatness of the forgetting curve, meaning that the shape of the curve is different for different users.
        - FSRS-6 default param.: FSRS-6 with default parameters.
        - FSRS-6 S0: FSRS-6 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
        - FSRS-6 binary: FSRS-6 which treats `hard` and `easy` grades as `good`.
        - FSRS-6 preset: different parameters are used for each preset. The minimum number of presets in Anki is one, a preset can be applied to multiple decks.
        - FSRS-6 deck: different parameters are used for each deck.
        - FSRS-6 recency: FSRS-6 trained with reviews being weighted based on their recency, such that older reviews affect the loss function less and newer reviews affect it more.
    - FSRS-rs: the Rust port of FSRS-6. See also: https://github.com/open-spaced-repetition/fsrs-rs
    - HLR: the algorithm proposed by Duolingo. Its full name is Half-Life Regression. For further information, please refer to the [this paper](https://github.com/duolingo/halflife-regression).
    - Ebisu v2: [an algorithm that uses Bayesian statistics](https://fasiha.github.io/ebisu/) to update its estimate of memory half-life after every review.

*In the two-component model of long-term memory, two independent variables are used to describe the status of unitary memory in a human brain: retrievability (R), or retrieval strength/probability of recall; and stability (S), or storage strength/memory half-life. The expanded three-component model adds a third variable - difficulty (D).

- Alternative models of memory:
    - DASH: the algorithm proposed in [this paper](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/zp38wc97m). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this algorithm: DASH[MCM] and DASH[ACT-R]. For further information, please refer to [this paper](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
    - ACT-R: the algorithm proposed in [this paper](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.

- Neural networks:
    - GRU: a type of recurrent neural network that's often used for making predictions based on a sequence of data. It's a classic in the field of machine learning for time-related tasks. It uses the same power forgetting curve as FSRS-4.5 and FSRS-5 to make the comparison more fair.
        - GRU-P: a variant of GRU that removes the fixed forgetting curve and predicts the probability of recall directly. This makes it more flexible than GRU, but also more prone to making strange predictions, such as the probability of recall *increasing* over time.
    - LSTM: a recurrent neural network with a more complex and sophisticated architecture than GRU. It is trained using the [Reptile algorithm](https://openai.com/index/reptile/). It uses short-term reviews and fractional intervals; enable the duration feature with `--duration` (disabled by default).
      The three aforementioned neural networks were first pretrained on 100 users and then further optimized on each user individually.
    - RWKV: uses a modified version of the [RWKV](https://github.com/BlinkDL/RWKV-LM) architecture, which combines the properties of an RNN and a Transformer. The neural network takes in as input the entire review history, all cards included. <a id="features-note"></a>Along with the usual features, it has additional access to: duration of the review, sibling card information, deck and preset structure/hierarchy, day of the week. Unlike other algorithms in this benchmark, RWKV is not optimized on each user individually. Instead, it is trained on 5 thousand users and evaluated on another 5 thousand; this process is repeated twice to get full coverage of the dataset.
        - RWKV-P: predicts the result of a review at the time just before the review. Does not have a forgetting curve in the traditional sense and predicts the probability of recall directly. Just like GRU-P, it may output unintuitive predictions, for example, it may never predict 100% or predict that the probability of recall will increase over time.
- Other:
    - AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline. An algorithm that doesn't outperform AVG cannot be considered good.
    - MOVING-AVG: unlike AVG, which uses the overall retention across all reviews as its prediction of probability of recall, MOVING-AVG outputs higher values if recent reviews were successful and lower values if recent reviews were lapses. By adjusting its predictions based on the outcome of recent reviews, it's trying to capture trends in retention instead of assuming that retention is always the same.
    - RMSE-BINS-EXPLOIT: an algorithm that exploits the calculation of RMSE (bins) by simulating the bins and keeping the error term close to 0.

For further information regarding the FSRS algorithm, please refer to the following wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of collections (each from one Anki user): 9,999.

Total number of reviews for evaluation: 349,923,850.
Same-day reviews are not used for evaluation, but some algorithms use them to refine their predictions of probability of recall for the next day. Some reviews are filtered out, for example, the revlog entries created by changing the due date manually or reviewing cards in a filtered deck with "Reschedule cards based on my answers in this deck" disabled. Finally, an outlier filter is applied. These are the reasons why the number of reviews used for evaluation is significantly lower than the figure of 727 million mentioned earlier. 

The following tables present the means and the 99% confidence intervals. The best result is highlighted in **bold**. The "Parameters" column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included. Arrows indicate whether lower (↓) or higher (↑) values are better.

For the sake of brevity, the following abbreviations are used in the "Input features" column:

**IL** = **i**nterval **l**engths, in days

**FIL** = **f**ractional (aka non-integer) **i**nterval **l**engths

**G** = **g**rades (Again/Hard/Good/Easy)

**SR** = **s**ame-day (or **s**hort-term) **r**eviews

**AT** = **a**nswer **t**ime (duration of the review), in milliseconds

### Without same-day reviews

| Algorithm | Parameters | Log Loss↓ | RMSE (bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| **RWKV-P** | 2762884 | **0.2773±0.0036** | 0.02502±0.00038 | **0.8329±0.0017** | [Yes](#features-note) |
| RWKV | 2762884 | 0.3193±0.0039 | 0.0540±0.0010 | 0.7683±0.0020 | [Yes](#features-note) |
| LSTM | 8869 | 0.3332±0.0041 | 0.05378±0.00096 | 0.7329±0.0020 | FIL, G, SR, AT |
| MOVING-AVG | 0 | 0.3369±0.0042 | 0.05915±0.00082 | 0.7001±0.0026 | --- |
| FSRS-rs | 21 | 0.3443±0.0041 | 0.0635±0.0011 | 0.7074±0.0022 | IL, G, SR |
| FSRS-6 recency | 21 | 0.3444±0.0042 | 0.0629±0.0010 | 0.7057±0.0023 | IL, G, SR |
| GRU-P-short | 297 | 0.3458±0.0043 | 0.0622±0.0011 | 0.6990±0.0025 | IL, G, SR|
| FSRS-6 | 21 | 0.3460±0.0042 | 0.0653±0.0011 | 0.7034±0.0023 | IL, G, SR |
| FSRS-6 preset | 21 | 0.3466±0.0043 | 0.0650±0.0011 | 0.7036±0.0023 | IL, G, SR |
| FSRS-6 binary | 17 | 0.3509±0.0043 | 0.0671±0.0011 | 0.6849±0.0024 | IL, G, SR |
| GRU-P | 297 | 0.3521±0.0043 | 0.0633±0.0011 | 0.6868±0.0025 | IL, G |
| FSRS-6 S0 | 4 | 0.3549±0.0042 | 0.0784±0.0012 | 0.6982±0.0023 | IL, G, SR |
| FSRS-6 deck | 21 | 0.3554±0.0044 | 0.0732±0.0013 | 0.6973±0.0023 | IL, G, SR |
| FSRS-5 | 19 | 0.3560±0.0045 | 0.0741±0.0013 | 0.7011±0.0023 | IL, G, SR |
| FSRS-4.5 | 17 | 0.3624±0.0046 | 0.0764±0.0013 | 0.6893±0.0023 | IL, G |
| FSRS-6 default param. | 0 | 0.3664±0.0044 | 0.0924±0.0014 | 0.6939±0.0023 | IL, G, SR |
| DASH-short | 9 | 0.3681±0.0045 | 0.0858±0.0014 | 0.6225±0.0029 | IL, G, SR|
| DASH | 9 | 0.3682±0.0045 | 0.0836±0.0013 | 0.6312±0.0026 | IL, G |
| DASH[MCM] | 9 | 0.3688±0.0045 | 0.0861±0.0014 | 0.6343±0.0026 | IL, G |
| FSRS v4 | 17 | 0.3726±0.0048 | 0.0838±0.0014 | 0.6853±0.0023 | IL, G |
| DASH[ACT-R] | 5 | 0.3728±0.0047 | 0.0886±0.0016 | 0.6239±0.0027 | IL, G |
| GRU | 39 | 0.3753±0.0047 | 0.0864±0.0013 | 0.6683±0.0023 | IL, G |
| AVG | 0 | 0.3945±0.0051 | 0.1034±0.0016 | 0.4997±0.0026 | --- |
| ACT-R | 5 | 0.4033±0.0054 | 0.1074±0.0017 | 0.5225±0.0025 | IL |
| FSRS v3 | 13 | 0.4364±0.0068 | 0.1097±0.0019 | 0.6605±0.0023 | IL, G |
| FSRS v2 | 14 | 0.4532±0.0072 | 0.1095±0.0020 | 0.6512±0.0023 | IL, G |
| HLR | 3 | 0.4694±0.0073 | 0.1275±0.0019 | 0.6369±0.0026 | IL, G |
| FSRS v1 | 7 | 0.4913±0.0079 | 0.1316±0.0023 | 0.6295±0.0025 | IL, G |
| HLR-short | 3 | 0.4929±0.0078 | 0.1397±0.0021 | 0.6115±0.0029 | IL, G, SR|
| Ebisu v2 | 0 | 0.4989±0.0078 | 0.1627±0.0022 | 0.6051±0.0025 | IL, G |
| **RMSE-BINS-EXPLOIT** | 0 | 4.608±0.067 | **0.01350±0.00027** | 0.6548±0.0022 | IL, G |

### With same-day reviews

Total number of collections: 10,000.

Total number of reviews for evaluation: 519,296,315.
Same-day reviews are used for evaluation. Here the probability of recall is calculated for all reviews, hence, the number of reviews for evaluation is greater.

| Model | Parameters | Log Loss↓ | RMSE(bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| **RWKV-P** | 2762884 | **0.2660±0.0036** | 0.03212±0.00045 | **0.8450±0.0017** | [Yes](#features-note) |
| RWKV | 2762884 | 0.2975±0.0037 | 0.05438±0.00081 | 0.7964±0.0017 | [Yes](#features-note) |
| LSTM | 8869 | 0.3140±0.0038 | 0.05200±0.00077 | 0.7622±0.0018 | FIL, G, SR, AT |
| MOVING-AVG | 0 | 0.3301±0.0044 | 0.0789±0.0010 | 0.7077±0.0024 | --- |
| DASH[MCM] | 9 | 0.3459±0.0042 | 0.0884±0.0011 | 0.6663±0.0025 | FIL, G, SR |
| GRU-P | 297 | 0.3487±0.0040 | 0.0838±0.0011 | 0.6457±0.0033 | FIL, G, SR |
| DASH | 9 | 0.3487±0.0041 | 0.0885±0.0011 | 0.6533±0.0027 | FIL, G, SR |
| DASH[ACT-R] | 5 | 0.3763±0.0045 | 0.1161±0.0014 | 0.5576±0.0030 | FIL, G, SR |
| FSRS-6 recency | 21 | 0.3790±0.0091 | 0.0847±0.0026 | 0.6855±0.0045 | FIL, G, SR |
| FSRS-6 | 21 | 0.3813±0.0092 | 0.0870±0.0026 | 0.6831±0.0045 | FIL, G, SR |
| AVG | 0 | 0.3816±0.0048 | 0.1195±0.0017 | 0.5006±0.0024 | --- |
| FSRS-6 binary | 17 | 0.3845±0.0090 | 0.0883±0.0027 | 0.6729±0.0045 | FIL, G, SR |
| FSRS-6 preset | 21 | 0.3863±0.0095 | 0.0864±0.0027 | 0.6821±0.0044 | FIL, G, SR |
| ACT-R | 5 | 0.3898±0.0049 | 0.1240±0.0017 | 0.5174±0.0028 | FIL, SR |
| FSRS-6 S0 | 4 | 0.417±0.011 | 0.1083±0.0031 | 0.6647±0.0042 | FIL, G, SR |
| FSRS-6 deck | 21 | 0.4285±0.0066 | 0.1052±0.0016 | 0.6709±0.0022 | FIL, G, SR |
| FSRS-4.5 | 17 | 0.4286±0.0060 | 0.1032±0.0014 | 0.6821±0.0023 | FIL, G, SR |
| FSRS-5 | 19 | 0.4565±0.0069 | 0.1175±0.0017 | 0.6761±0.0023 | FIL, G, SR |
| FSRS-6 default param. | 0 | 0.482±0.015 | 0.1195±0.0037 | 0.6432±0.0042 | FIL, G, SR |
| FSRS v4 | 17 | 0.4848±0.0077 | 0.1159±0.0017 | 0.6663±0.0024 | FIL, G, SR |
| GRU | 39 | 0.590±0.010 | 0.1846±0.0027 | 0.5984±0.0031 | FIL, G, SR |
| FSRS v3 | 13 | 0.647±0.012 | 0.1413±0.0021 | 0.6389±0.0025 | FIL, G, SR |
| FSRS v2 | 14 | 0.663±0.012 | 0.1351±0.0019 | 0.6386±0.0023 | FIL, G, SR |
| HLR | 3 | 0.705±0.014 | 0.1715±0.0024 | 0.6104±0.0028 | FIL, G, SR |
| FSRS v1 | 7 | 0.744±0.015 | 0.1584±0.0023 | 0.6080±0.0026 | FIL, G, SR |
| Ebisu v2 | 0 | 0.772±0.017 | 0.1846±0.0026 | 0.5947±0.0031 | FIL, G, SR |
| **RMSE-BINS-EXPLOIT** | 0 | 4.129±0.059 | **0.01547±0.00026** | 0.6761±0.0021 | FIL, G, SR |

### Superiority

The metrics presented above can be difficult to interpret. In order to make it easier to understand how algorithms perform relative to each other, the images below shows the percentage of users for whom algorithm A (row) has a lower Log Loss than algorithm B (column).

#### Without same-day reviews

This table is based on 9,999 collections. To make the table easier to read, not all the algorithms are included.

![Superiority, 9999](./plots/Superiority-small-9999-collections.png)

Additionally, you can find the full table [here](./plots/Superiority-9999.png).

#### With same-day reviews

This table is based on 10,000 collections.

![Superiority, 10000](./plots/Superiority-small-10000-collections.png)

## Default Parameters

FSRS-6:

```
0.212, 1.2931, 2.3065, 8.2956,
6.4133, 0.8334, 3.0194, 0.001,
1.8722, 0.1666, 0.796,
1.4835, 0.0614, 0.2629, 1.6483,
0.6014, 1.8729,
0.5425, 0.0912, 0.0658,
0.1542,
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
uv sync
```

> uv is a tool that helps manage Python environments and dependencies. You can install it from https://docs.astral.sh/uv/.

### Commands

FSRS-6:

```bash
uv run script.py --algo FSRS-6 --short
```

FSRS-6 with default parameters:

```bash
uv run script.py --algo FSRS-6 --default --short
```

FSRS-6 with only the first 4 parameters optimized:

```bash
uv run script.py --algo FSRS-6 --S0 --short
```

FSRS-rs:

Then run the following command:

```bash
uv run script.py --algo FSRS-rs --short
```

> Please place the [fsrs-optimizer repository](https://github.com/open-spaced-repetition/fsrs-optimizer) in the same directory as this repository.

Set the number of processes:

```bash
uv run script.py --processes 4
```

Save the raw predictions:

```bash
uv run script.py --raw
```

Save the detailed results:

```bash
uv run script.py --file
```

Save the analyzing charts:

```bash
uv run script.py --plot
```

Benchmark FSRS-5/FSRSv4/FSRSv3/HLR/LSTM/SM2:

```bash
uv run script.py --algo FSRS-6
```

> You can change `FSRS-6` to `FSRSv3`, `HLR`, `LSTM`, etc. to run the corresponding algorithm.

Instead of using a 5-way split, train the algorithm and evaluate it on the same data. This can be useful to determine how much the algorithm is overfitting.
```bash
uv run script.py --algo FSRS-6 --train_equals_test
```

> You can limit the number of users with the --max-user-id parameter. Only users from 1 to max-user-id included will be considered.
```bash
uv run script.py --max-user-id 20
```

### script.py options

Run `uv run script.py --help` for the full list. Common options include:

| Flag | Description | Default/Notes |
| --- | --- | --- |
| `--algo` | Algorithm name (e.g., `FSRS-6`, `GRU`, `LSTM`). | Default: `FSRSv3` |
| `--short` | Include short-term (same-day) reviews. | Off |
| `--secs` | Use `elapsed_seconds` as the interval instead of days. | Off |
| `--duration` | Add review duration feature (LSTM only). | Off |
| `--default` | Evaluate default parameters (no training). | Off |
| `--S0` | FSRS-5/FSRS-6 with only S0 initialization. | Off |
| `--two_buttons` | Treat Hard and Easy as Good. | Off |
| `--recency` | Enable recency weighting during training. | Off |
| `--partitions` | Partition training by `deck` or `preset`. | Default: `none` |
| `--data` | Path to `revlogs/*.parquet`. | Default: `../anki-revlogs-10k` |
| `--processes` | Number of worker processes. | Default: `8` |
| `--max-user-id` | Maximum user ID to process (inclusive). | No limit |
| `--n_splits` | Number of TimeSeriesSplit folds. | Default: `5` |
| `--train_equals_test` | Train and test on the same data. | Off |
| `--no_test_same_day` | Exclude elapsed_days=0 from the test set. | Off |
| `--no_train_same_day` | Exclude elapsed_days=0 from the train set. | Off |
| `--equalize_test_with_non_secs` | Test only on reviews included in non-secs tests. | Off |
| `--batch_size` | Batch size for training models. | Default: `512` |
| `--max_seq_len` | Max sequence length for batching inputs. | Default: `64` |
| `--torch_num_threads` | PyTorch intra-op threads. | Default: `1` |
| `--raw` | Save raw predictions to `raw/<name>.jsonl`. | Off |
| `--file` | Save per-user evaluation TSVs to `evaluation/<name>/`. | Off |
| `--plot` | Save evaluation plots to `evaluation/<name>/` (if available). | Off |
| `--weights` | Save model weights to `weights/<name>/`. | Off |
| `--dev` | Enable local dev import for `fsrs_optimizer`. | Off |

## Pretrain

To pretrain LSTM on multiple users, run:

```bash
uv run pretrain.py --algo LSTM
```