import unittest
import pandas as pd
import numpy as np
from ..evaluator import (
    calculate_mae,
    calculate_mse,
    calculate_rmse,
    calculate_mape,
    calculate_smape,
    evaluate_all_metrics
)

class TestEvaluatorMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true_basic = pd.Series([10, 20, 30, 40, 50])
        self.y_pred_basic = pd.Series([12, 18, 33, 38, 52])

        self.y_true_with_zeros = pd.Series([0, 10, 20, 0, 30])
        self.y_pred_with_zeros = pd.Series([0, 12, 18, 2, 28])

        self.y_true_all_zeros = pd.Series([0, 0, 0])
        self.y_pred_some_zeros = pd.Series([0, 1, 0])

        self.y_true_empty = pd.Series([], dtype=float)
        self.y_pred_empty = pd.Series([], dtype=float)

    def test_calculate_mae(self):
        self.assertAlmostEqual(calculate_mae(self.y_true_basic, self.y_pred_basic), 2.0) # (2+2+3+2+2)/5 = 11/5 = 2.2
        # Corrected expected MAE: (|10-12|+|20-18|+|30-33|+|40-38|+|50-52|) / 5 = (2+2+3+2+2)/5 = 11/5 = 2.2
        self.assertAlmostEqual(calculate_mae(self.y_true_basic, self.y_pred_basic), 2.2)


    def test_calculate_mse(self):
        # MSE: (2^2 + (-2)^2 + 3^2 + (-2)^2 + 2^2) / 5 = (4+4+9+4+4)/5 = 25/5 = 5.0
        self.assertAlmostEqual(calculate_mse(self.y_true_basic, self.y_pred_basic), 5.0)

    def test_calculate_rmse(self):
        self.assertAlmostEqual(calculate_rmse(self.y_true_basic, self.y_pred_basic), np.sqrt(5.0))

    def test_calculate_mape(self):
        # MAPE for basic:
        # (|2/10| + |2/20| + |3/30| + |2/40| + |2/50|) / 5 * 100
        # = (0.2 + 0.1 + 0.1 + 0.05 + 0.04) / 5 * 100
        # = (0.49) / 5 * 100 = 0.098 * 100 = 9.8%
        self.assertAlmostEqual(calculate_mape(self.y_true_basic, self.y_pred_basic), 9.8)

        # MAPE with zeros in true values (zeros in y_true are excluded from calculation)
        # y_true_nz = [10, 20, 30], y_pred_nz = [12, 18, 28] (corresponding to non-zero y_true)
        # Errors: |(10-12)/10|, |(20-18)/20|, |(30-28)/30|
        #         = | -2/10 |, |  2/20 |, |  2/30 |
        #         =   0.2,       0.1,     0.06666...
        # Mean = (0.2 + 0.1 + 0.06666666666666667) / 3 * 100 = 0.3666666666666667 / 3 * 100 = 0.12222222222 * 100 = 12.222...%
        self.assertAlmostEqual(calculate_mape(self.y_true_with_zeros, self.y_pred_with_zeros), 12.222222222222221)

        self.assertTrue(np.isnan(calculate_mape(self.y_true_all_zeros, self.y_pred_some_zeros)))

    def test_calculate_smape(self):
        # sMAPE for basic:
        # Sum(|pred-true| / ((|true|+|pred|)/2)) / N * 100
        # 1: |2| / ((10+12)/2) = 2/11 = 0.181818
        # 2: |2| / ((20+18)/2) = 2/19 = 0.105263
        # 3: |3| / ((30+33)/2) = 3/31.5 = 0.095238
        # 4: |2| / ((40+38)/2) = 2/39 = 0.051282
        # 5: |2| / ((50+52)/2) = 2/51 = 0.039215
        # Sum = 0.181818 + 0.105263 + 0.095238 + 0.051282 + 0.039215 = 0.472816
        # Mean = 0.472816 / 5 = 0.0945632
        # Result = 0.0945632 * 100 = 9.45632 %
        self.assertAlmostEqual(calculate_smape(self.y_true_basic, self.y_pred_basic), 9.456321936743144)

        # sMAPE with zeros:
        # y_true_with_zeros = pd.Series([0, 10, 20, 0, 30])
        # y_pred_with_zeros = pd.Series([0, 12, 18, 2, 28])
        # 1: |0-0| / ((0+0)/2) -> num=0, den=0. My code treats this as 0.
        # 2: |12-10| / ((10+12)/2) = 2/11 = 0.181818
        # 3: |18-20| / ((20+18)/2) = 2/19 = 0.105263
        # 4: |2-0| / ((0+2)/2) = 2/1 = 2.0  -> This term is large due to true zero.
        # 5: |28-30| / ((30+28)/2) = 2/29 = 0.068965
        # Sum = 0 + 0.181818 + 0.105263 + 2.0 + 0.068965 = 2.356046
        # Mean = 2.356046 / 5 = 0.4712092
        # Result = 0.4712092 * 100 = 47.12092 %
        self.assertAlmostEqual(calculate_smape(self.y_true_with_zeros, self.y_pred_with_zeros), 47.12092483941852)


    def test_evaluate_all_metrics(self):
        metrics = evaluate_all_metrics(self.y_true_basic, self.y_pred_basic)
        self.assertAlmostEqual(metrics['mae'], 2.2)
        self.assertAlmostEqual(metrics['mse'], 5.0)
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(5.0))
        self.assertAlmostEqual(metrics['mape'], 9.8)
        self.assertAlmostEqual(metrics['smape'], 9.456321936743144)

    def test_evaluate_all_metrics_empty(self):
        metrics = evaluate_all_metrics(self.y_true_empty, self.y_pred_empty)
        for metric_name in metrics:
            self.assertTrue(np.isnan(metrics[metric_name]), f"{metric_name} should be NaN for empty series.")

    def test_length_mismatch(self):
        y_short = pd.Series([1,2])
        with self.assertRaisesRegex(ValueError, "y_true and y_pred must have the same length."):
            evaluate_all_metrics(self.y_true_basic, y_short)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# To run these tests from the `backend` directory:
# python -m app.tests.test_evaluator
# Or, if using a test runner like pytest (after installing it):
# pytest
```
