# FSRS-7 Ordinal Experiment

## Purpose

`FSRS-7-ordinal` is an experimental benchmark model that keeps FSRS-7's recall
prediction as the reported output while adding a training-time ordinal button
likelihood for ratings 1-4.

The final benchmark `.jsonl` metrics remain based on the usual binary recall
probability `R`, so `LogLoss` in result files is still pure recall log loss.

## Data Flow

- Data read:
  - existing FSRS tensor history from `tensor`
  - current review interval from `delta_t`
  - binary recall label from `y`
  - current button from `rating`
  - optional row weight from `weights`
- Data created:
  - in-memory training batches that include the current `rating` as an extra
    target tensor
  - in-memory four-button probabilities for the auxiliary ordinal loss
- Data updated:
  - FSRS-7 parameters through the existing recall loss
  - six ordinal parameters `a0..a5` through the auxiliary ordinal loss
- Data storage:
  - no new persistent training data
  - saved benchmark predictions still store only binary recall predictions

## Ordinal Probability Model

For a review state with retrievability `R`, stability `S`, and difficulty `D`:

```text
raw = a0 + a1 * logit(R) + a2 * log(S) + a3 * D

H = sigmoid(raw - a4)
E = sigmoid(raw - (a4 + softplus(a5)))
```

The `softplus(a5)` offset guarantees the Easy threshold is above the Hard/Good
threshold, so `E <= H`.

Button probabilities:

```text
P(Again) = 1 - R
P(Hard)  = R * (1 - H)
P(Good)  = R * (H - E)
P(Easy)  = R * E
```

These probabilities are non-negative and sum to 1.0.

## Training And Evaluation

Training uses the existing binary recall BCE plus the auxiliary four-button
negative log likelihood:

```text
ordinal_loss = -log(P(observed_button))
```

Evaluation and result `.jsonl` output use only the binary recall prediction
`R`, without adding the ordinal term or scheduling penalties to the reported
metrics.

Use `evaluate.py --models ... --max-user-id ...` to compare this experiment
against a baseline on the same user subset. The evaluator intersects users
across the selected result files before computing aggregate metrics.
