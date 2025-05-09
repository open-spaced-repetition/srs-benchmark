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

### Metrics

We use three metrics in the SRS benchmark to evaluate how well these algorithms work: Log Loss, AUC, and a custom RMSE that we call RMSE (bins).

- Log Loss (also known as Binary Cross Entropy): used primarily in binary classification problems, Log Loss serves as a measure of the discrepancies between predicted probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm approximates the true recall probabilities. Log Loss ranges from 0 to infinity, lower is better.
- Root Mean Square Error in Bins (RMSE (bins)): this is a metric designed for use in the SRS benchmark. In this approach, predictions and review outcomes are grouped into bins based on three features: the interval length, the number of reviews, and the number of lapses. Within each bin, the squared difference between the average predicted probability of recall and the average recall rate is calculated. These values are then weighted according to the sample size in each bin, and then the final weighted root mean square error is calculated. This metric provides a nuanced understanding of algorithm performance across different probability ranges. For more details, you can read [The Metric](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Metric). RMSE (bins) ranges from 0 to 1, lower is better.
- AUC (Area under the ROC Curve): this metric tells us how much the algorithm is capable of distinguishing between classes. AUC ranges from 0 to 1, however, in practice it's almost always greater than 0.5; higher is better.

Log Loss and RMSE (bins) measure calibration: how well predicted probabilities of recall match the real data. AUC measures discrimination: how well the algorithm can tell two (or more, generally speaking) classes apart. AUC can be good (high) even if Log Loss and RMSE are poor.

### Algorithms and algorithm families

- Two component or three component* model of memory:
    - FSRS v1 and v2: the initial experimental versions of FSRS.
    - FSRS v3: the first official release of the FSRS algorithm, made available as a custom scheduling script.
    - FSRS v4: the upgraded version of FSRS, made better with help from the community.
    - FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed.
    - FSRS-5: the upgraded version of FSRS. Unlike the previous versions, it uses the same-day review data. Same-day reviews are used only for training, and not for evaluation.
    - FSRS-6: the latest version of FSRS. The formula for handling same-day reviews has been improved. More importantly, FSRS-6 has an optimizable parameter that controls the flatness of the forgetting curve, meaning that the shape of the curve is different for different users.
        - FSRS-6 default param.: FSRS-6 with default parameters. The default parameters have been obtained by running FSRS-6 on all 10 thousand collections from the dataset and calculating the median of each parameter.
        - FSRS-6 pretrain: FSRS-6 where only the first 4 parameters (values of initial stability after the first review) are optimized and the rest are set to default.
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
    - LSTM: a recurrent neural network with a more complex and sophisticated architecture than GRU. It is trained using the [Reptile algorithm](https://openai.com/index/reptile/). It uses short-term reviews, fractional intervals, and the duration of review as part of its input.
      The three aforementioned neural networks were first pretrained on 100 users and then further optimized on each user individually.
    - NN-17: a neural network approximation of [SM-17](https://supermemo.guru/wiki/Algorithm_SM-17). It has a comparable number of parameters, and according to our estimates, it performs similarly to SM-17.

- SM-2-based algorithms:
    - SM-2: one of the early algorithms used by SuperMemo, the first spaced repetition software. It was developed more than 30 years ago, and it's still popular today. [Anki's default algorithm is based on SM-2](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html), [Mnemosyne](https://mnemosyne-proj.org/principles.php) also uses it. This algorithm does not predict the probability of recall natively; therefore, for the sake of the benchmark, the output was modified based on some assumptions about the forgetting curve. The algorithm is described by Piotr Wozniak [here](https://super-memory.com/english/ol/sm2.htm).
        - SM-2 trainable: SM-2 algorithm with optimizable parameters.
    - Anki-SM-2: a variant of the SM-2 algorithm that is used in Anki.
        - Anki-SM-2 trainable: Anki algorithm with optimizable parameters.

- Other:
    - AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline.
    - RMSE-BINS-EXPLOIT: an algorithm that exploits the calculation of RMSE (bins) by simulating the bins and keeping the error term close to 0.

For further information regarding the FSRS algorithm, please refer to the following wiki page: [The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Result

Total number of users: 9,999.

Total number of reviews for evaluation: 349,923,850.
Same-day reviews are not used for evaluation, but some algorithms use them to refine their predictions of probability of recall for the next day. Some reviews are filtered out, for example, the revlog entries created by changing the due date manually or reviewing cards in a filtered deck with "Reschedule cards based on my answers in this deck" disabled. Finally, an outlier filter is applied. These are the reasons why the number of reviews used for evaluation is significantly lower than the figure of 727 million mentioned earlier. 

The following tables present the means and the 99% confidence intervals. The best result is highlighted in **bold**. The "Parameters" column shows the number of optimizable (trainable) parameters. If a parameter is a constant, it is not included. Arrows indicate whether lower (↓) or higher (↑) values are better.

For the sake of brevity, the following abbreviations are used in the "Input features" column:

**IL** = **i**nterval **l**engths, in days

**FIL** = **f**ractional (aka non-integer) **i**nterval **l**engths

**G** = **g**rades (Again/Hard/Good/Easy)

**SR** = **s**ame-day (or **s**hort-term) **r**eviews

**AT** = **a**nswer **t**ime (duration of the review), in milliseconds

### Weighted by the number of reviews

| Algorithm | Parameters | Log Loss↓ | RMSE (bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| **LSTM** | 8869 | **0.312±0.0078** | 0.035±0.0011 | **0.733±0.0038** | FIL, G, SR, AT |
| GRU-P-short | 297 | 0.320±0.0080 | 0.042±0.0013 | 0.710±0.0047 | IL, G, SR|
| FSRS-6 recency | 21 | 0.320±0.0081 | 0.044±0.0013 | 0.710±0.0040 | IL, G, SR |
| FSRS-rs | 21 | 0.320±0.0082 | 0.044±0.0012 | 0.709±0.0041 | IL, G, SR |
| FSRS-6 | 21 | 0.321±0.0083 | 0.046±0.0013 | 0.706±0.0041 | IL, G, SR |
| FSRS-6 preset | 21 | 0.322±0.0081 | 0.046±0.0013 | 0.707±0.0041 | IL, G, SR |
| GRU-P | 297 | 0.325±0.0081 | 0.043±0.0013 | 0.699±0.0046 | IL, G |
| FSRS-6 binary | 17 | 0.326±0.0081 | 0.049±0.0013 | 0.686±0.0047 | IL, G, SR |
| FSRS-5 | 19 | 0.327±0.0083 | 0.052±0.0015 | 0.702±0.0042 | IL, G, SR |
| FSRS-6 deck | 21 | 0.329±0.0082 | 0.052±0.0016 | 0.699±0.0041 | IL, G, SR |
| FSRS-4.5 | 17 | 0.332±0.0083 | 0.054±0.0016 | 0.692±0.0041 | IL, G |
| FSRS v4 | 17 | 0.338±0.0086 | 0.058±0.0017 | 0.689±0.0043 | IL, G |
| DASH-short | 9 | 0.339±0.0084 | 0.066±0.0019 | 0.636±0.0050 | IL, G, SR |
| FSRS-6 pretrain | 4 | 0.339±0.0084 | 0.070±0.0024 | 0.695±0.0039 | IL, G, SR |
| DASH | 9 | 0.340±0.0086 | 0.063±0.0017 | 0.639±0.0046 | IL, G |
| DASH[MCM] | 9 | 0.340±0.0085 | 0.064±0.0018 | 0.640±0.0051 | IL, G |
| GRU | 39 | 0.343±0.0088 | 0.063±0.0017 | 0.673±0.0039 | IL, G |
| DASH[ACT-R] | 5 | 0.343±0.0087 | 0.067±0.0019 | 0.629±0.0049 | IL, G |
| FSRS-6 default param. | 0 | 0.347±0.0087 | 0.079±0.0027 | 0.692±0.0041 | IL, G, SR |
| ACT-R | 5 | 0.362±0.0089 | 0.086±0.0024 | 0.534±0.0054 | IL |
| AVG | 0 | 0.363±0.0090 | 0.088±0.0025 | 0.508±0.0046 | --- |
| FSRS v3 | 13 | 0.371±0.0099 | 0.073±0.0021 | 0.667±0.0047 | IL, G |
| FSRS v2 | 14 | 0.38±0.010 | 0.069±0.0021 | 0.667±0.0048 | IL, G |
| NN-17 | 39 | 0.38±0.027 | 0.081±0.0038 | 0.611±0.0043 | IL, G |
| FSRS v1 | 7 | 0.40±0.011 | 0.086±0.0024 | 0.633±0.0046 | IL, G |
| Anki-SM-2 trainable | 7 | 0.41±0.011 | 0.094±0.0030 | 0.616±0.0057 | IL, G |
| HLR | 3 | 0.41±0.012 | 0.105±0.0030 | 0.633±0.0050 | IL, G |
| HLR-short | 3 | 0.44±0.013 | 0.116±0.0036 | 0.615±0.0062 | IL, G, SR |
| SM-2 trainable | 6 | 0.44±0.012 | 0.119±0.0033 | 0.599±0.0050 | IL, G |
| Ebisu v2 | 0 | 0.46±0.012 | 0.158±0.0038 | 0.594±0.0050 | IL, G |
| Anki-SM-2 | 0 | 0.49±0.015 | 0.128±0.0037 | 0.597±0.0055 | IL, G |
| SM-2-short | 0 | 0.51±0.015 | 0.128±0.0038 | 0.593±0.0064 | IL, G, SR |
| SM-2 | 0 | 0.55±0.017 | 0.148±0.0041 | 0.600±0.0051 | IL, G |
| **RMSE-BINS-EXPLOIT** | 0 | 4.5±0.13 | **0.0062±0.00022** | 0.638±0.0040 | IL, G |

### Unweighted

| Algorithm | Parameters | Log Loss↓ | RMSE (bins)↓ | AUC↑ | Input features |
| --- | --- | --- | --- | --- | --- |
| **LSTM** | 8869 | **0.333±0.0042** | 0.0538±0.00096 | **0.733±0.0021** | FIL, G, SR, AT |
| FSRS-6 recency | 21 | 0.344±0.0041 | 0.063±0.0010 | 0.710±0.0023 | IL, G, SR |
| FSRS-rs | 21 | 0.344±0.0041 | 0.063±0.0010 | 0.710±0.0022 | IL, G, SR |
| FSRS-6 | 21 | 0.345±0.0042 | 0.066±0.0011 | 0.707±0.0023 | IL, G, SR |
| GRU-P-short | 297 | 0.346±0.0042 | 0.062±0.0011 | 0.699±0.0026 | IL, G, SR|
| FSRS-6 preset | 21 | 0.346±0.0042 | 0.065±0.0010 | 0.708±0.0023 | IL, G, SR |
| FSRS-6 binary | 17 | 0.351±0.0043 | 0.068±0.0011 | 0.685±0.0025 | IL, G, SR |
| GRU-P | 297 | 0.352±0.0042 | 0.063±0.0011 | 0.687±0.0025 | IL, G |
| FSRS-6 deck | 21 | 0.355±0.0045 | 0.074±0.0013 | 0.703±0.0023 | IL, G, SR |
| FSRS-5 | 19 | 0.356±0.0043 | 0.074±0.0012 | 0.701±0.0023 | IL, G, SR |
| FSRS-6 pretrain | 4 | 0.359±0.0044 | 0.083±0.0013 | 0.702±0.0022 | IL, G, SR |
| FSRS-4.5 | 17 | 0.362±0.0045 | 0.076±0.0013 | 0.689±0.0023 | IL, G |
| DASH | 9 | 0.368±0.0045 | 0.084±0.0013 | 0.631±0.0027 | IL, G |
| DASH-short | 9 | 0.368±0.0045 | 0.086±0.0014 | 0.622±0.0029 | IL, G, SR|
| DASH[MCM] | 9 | 0.369±0.0044 | 0.086±0.0014 | 0.634±0.0026 | IL, G |
| FSRS-6 default param. | 0 | 0.371±0.0046 | 0.097±0.0015 | 0.701±0.0022 | IL, G, SR |
| FSRS v4 | 17 | 0.373±0.0048 | 0.084±0.0014 | 0.685±0.0023 | IL, G |
| DASH[ACT-R] | 5 | 0.373±0.0047 | 0.089±0.0016 | 0.624±0.0027 | IL, G |
| GRU | 39 | 0.375±0.0047 | 0.086±0.0014 | 0.668±0.0023 | IL, G |
| AVG | 0 | 0.394±0.0050 | 0.103±0.0016 | 0.500±0.0026 | --- |
| NN-17 | 39 | 0.398±0.0049 | 0.101±0.0013 | 0.624±0.0023 | IL, G |
| ACT-R | 5 | 0.403±0.0055 | 0.107±0.0017 | 0.522±0.0024 | IL |
| FSRS v3 | 13 | 0.436±0.0067 | 0.110±0.0020 | 0.661±0.0024 | IL, G |
| FSRS v2 | 14 | 0.453±0.0072 | 0.110±0.0020 | 0.651±0.0023 | IL, G |
| HLR | 3 | 0.469±0.0073 | 0.128±0.0019 | 0.637±0.0026 | IL, G |
| FSRS v1 | 7 | 0.491±0.0080 | 0.132±0.0022 | 0.630±0.0025 | IL, G |
| HLR-short | 3 | 0.493±0.0079 | 0.140±0.0021 | 0.611±0.0029 | IL, G, SR|
| Ebisu v2 | 0 | 0.499±0.0078 | 0.163±0.0021 | 0.605±0.0026 | IL, G |
| Anki-SM-2 trainable | 7 | 0.513±0.0089 | 0.140±0.0024 | 0.618±0.0023 | IL, G |
| SM-2 trainable | 6 | 0.58±0.012 | 0.170±0.0028 | 0.597±0.0025 | IL, G |
| Anki-SM-2 | 0 | 0.62±0.011 | 0.172±0.0026 | 0.613±0.0022 | IL, G |
| SM-2-short | 0 | 0.65±0.015 | 0.170±0.0028 | 0.590±0.0027 | IL, G, SR|
| SM-2 | 0 | 0.72±0.017 | 0.203±0.0030 | 0.603±0.0025 | IL, G |
| **RMSE-BINS-EXPLOIT** | 0 | 4.61±0.067 | **0.0135±0.00028** | 0.655±0.0021 | IL, G |

Averages weighted by the number of reviews are more representative of "best case" performance when plenty of data is available. Since almost all algorithms perform better when there's a lot of data to learn from, weighting by n(reviews) biases the average towards lower values.

Unweighted averages are more representative of "average case" performance. In reality, not every user will have hundreds of thousands of reviews, so the algorithm won't always be able to reach its full potential.

### Superiority

The metrics presented above can be difficult to interpret. In order to make it easier to understand how algorithms perform relative to each other, the image below shows the percentage of users for whom algorithm A (row) has a lower Log Loss than algorithm B (column). For example, FSRS-6-recency has a 99.6% superiority over the Anki's variant of SM-2 with default parameters, meaning that for 99.6% of all collections in this benchmark, FSRS-6-recency can estimate the probability of recall more accurately. However, please keep in mind that SM-2 wasn't designed to predict probabilities, and the only reason it does so in this benchmark is because extra formulae have been added to it.

This table is based on 9,999 collections. To make the table easier to read, not all the algorithms are included.

![Superiority, 9999](./plots/Superiority-small-9999-collections.png)

Additionally, you can find the full table [here](./plots/Superiority-9999.png).

### Statistical significance

The figures below show effect sizes comparing the Log Loss between all pairs of algorithms using the Wilcoxon signed-rank test r-values:

The colors indicate:

- Red shades indicate the row algorithm performs worse than the column algorithm:
  - Dark red: large effect (r > 0.5)
  - Red: medium effect (0.5 ≥ r > 0.2) 
  - Light red: small effect (r ≤ 0.2)

- Green shades indicate the row algorithm performs better than the column algorithm:
  - Dark green: large effect (r > 0.5)
  - Green: medium effect (0.5 ≥ r > 0.2) 
  - Light green: small effect (r ≤ 0.2)

- Grey indicates that the p-value is greater than 0.01, meaning we cannot conclude which algorithm performs better.

The Wilcoxon test considers both the sign and rank of differences between pairs, but it does not account for the varying number of reviews across collections. Therefore, while the test results are reliable for qualitative analysis, caution should be exercised when interpreting the specific magnitude of effects.

To make the table easier to read, not all the algorithms are included.

![Wilcoxon, 9999 collections](./plots/Wilcoxon-small-9999-collections.png)

Additionally, you can find the full table [here](./plots/Wilcoxon-9999-collections.png).

## Default Parameters

FSRS-6:

```
0.2172, 1.1771, 3.2602, 16.1507,
7.0114, 0.57, 2.0966, 0.0069,
1.5261, 0.112, 1.0178,
1.849, 0.1133, 0.3127, 2.2934,
0.2191, 3.0004,
0.7536, 0.3332, 0.1437,
0.2,
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

FSRS-6:

```bash
python script.py
```

> Please confirm that you have upgraded the `fsrs-optimizer` package to the latest version.

FSRS-6 with default parameters:

```bash
python script.py --dry
```

FSRS-6 with only the first 4 parameters optimized:

```bash
python script.py --pretrain
```

FSRS-rs:

It requires `fsrs_rs_python` to be installed.

```bash
pip install fsrs_rs_python
```

Then run the following command:

```bash
python script.py --rust
```

Dev algorithm in fsrs-optimizer:

```bash
python script.py --dev
```

> Please place the [fsrs-optimizer repository](https://github.com/open-spaced-repetition/fsrs-optimizer) in the same directory as this repository.

Set the number of processes:

```bash
python script.py --processes 4
```

Save the raw predictions:

```bash
python script.py --raw
```

Save the detailed results:

```bash
python script.py --file
```

Save the analyzing charts:

```bash
python script.py --plot
```

Benchmark FSRS-5/FSRSv4/FSRSv3/HLR/LSTM/SM2:

```bash
python other.py --algo FSRS-6
```

> You can change `FSRS-6` to `FSRSv3`, `HLR`, `LSTM`, etc. to run the corresponding algorithm.

Instead of using a 5-way split, train the algorithm and evaluate it on the same data. This can be useful to determine how much the algorithm is overfitting.
```bash
python other.py --algo FSRS-6 --train_equals_test
```
