Two-tower vs baselines, test split (108,910 reviews, 50,522 users). delta = baseline - two-tower, so positive means the two-tower is better. 95% CIs and p-values come from a user-clustered bootstrap (1000 resamples); p-values are Holm-adjusted across the 6 baselines. The smallest p a bootstrap this size can give is about 0.001 per comparison, so a value at that floor means no resample crossed zero.

|                 |   rmse (baseline) |   rmse (two-tower) |   delta rmse | 95% CI             |   p (cluster boot) |   delta mae | 95% CI (mae)       |   p (mae) |
|:----------------|------------------:|-------------------:|-------------:|:-------------------|-------------------:|------------:|:-------------------|----------:|
| user+item as-of |             1.042 |              1.026 |      0.01638 | [+0.0148, +0.0179] |           0.005994 |    -0.0438  | [-0.0457, -0.0420] |  0.005994 |
| user+item bias  |             1.078 |              1.026 |      0.05154 | [+0.0491, +0.0539] |           0.005994 |     0.07005 | [+0.0671, +0.0734] |  0.005994 |
| user mean       |             1.08  |              1.026 |      0.0535  | [+0.0503, +0.0567] |           0.005994 |     0.068   | [+0.0646, +0.0714] |  0.005994 |
| item mean       |             1.104 |              1.026 |      0.07795 | [+0.0751, +0.0810] |           0.005994 |     0.1178  | [+0.1136, +0.1225] |  0.005994 |
| item kNN (k=40) |             1.108 |              1.026 |      0.08172 | [+0.0789, +0.0846] |           0.005994 |     0.1178  | [+0.1135, +0.1224] |  0.005994 |
| global mean     |             1.112 |              1.026 |      0.08553 | [+0.0821, +0.0893] |           0.005994 |     0.1325  | [+0.1281, +0.1373] |  0.005994 |

Strongest baseline (`user+item as-of`) by user history:

|                               |   rmse (baseline) |   rmse (two-tower) |   delta rmse | 95% CI             |   p (cluster boot) |   delta mae | 95% CI (mae)       |   p (mae) |
|:------------------------------|------------------:|-------------------:|-------------:|:-------------------|-------------------:|------------:|:-------------------|----------:|
| user+item as-of - no history  |            1.296  |             1.274  |      0.02222 | [+0.0201, +0.0243] |           0.000999 |    -0.0155  | [-0.0175, -0.0135] |  0.000999 |
| user+item as-of - has history |            0.8578 |             0.8459 |      0.01182 | [+0.0097, +0.0141] |           0.000999 |    -0.06056 | [-0.0629, -0.0581] |  0.000999 |
