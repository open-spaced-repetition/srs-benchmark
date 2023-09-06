# FSRS Benchmark

## Result

Total number of users: 71

Total number of reviews for evaluation: 4,632,965

### Weighted by number of reviews

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3879 | 0.3346 | 0.0453 |
| LSTM | 0.4199 | 0.3425 | 0.0662 |
| FSRS v3 | 0.4890 | 0.3633 | 0.1204 |

### Weighted by ln(number of reviews)

| Algorithm | Log Loss | RMSE | RMSE(bins) |
| --- | --- | --- | --- |
| FSRS v4 | 0.3830 | 0.3311 | 0.0543 |
| FSRS v3 | 0.5132 | 0.3670 | 0.1326 |
| LSTM | 0.5788 | 0.3752 | 0.1385 |