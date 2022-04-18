# Sea Ice Extent Forecasting using Convolutional LSTMs
<!-- Term project for SYDE 675: Sea ice extent forecasting using LSTMs, attention, and multiple timeframes -->

## Abstract
In this paper, a data-driven approach is taken to predict Arctic sea ice extent (SIE). We first assess the work of (Ali et al., 2022) in their paper ”Sea Ice Forecasting using Attention-based Ensemble LSTM”. We find that their results are not perfectly reproducible, and that their model does not outperform a naive statistical baseline. We then propose an alternative model architecture which accepts both spatial and temporal inputs, and uses several new climatic input variables. We test a variety of hyperparameters and find our best model outperforming a baseline statistical model by 56% in %RMSE.

## Folder Structure
```
.
├── Data
│   ├── Arctic_domain_mean_1979_2018.csv
│   ├── Arctic_domain_mean_monthly_1979_2018.csv
│   ├── dailyt30_features.npy
│   ├── dailyt30_target.npy
│   ├── extents.nc
│   ├── monthly_features.npy
│   ├── monthly_target.npy
│   └── X_grid.npy
├── Models
│   ├── Improvements
│   │   ├── evaluation.ipynb
│   │   ├── model.py
│   │   ├── pre-process.ipynb
│   │   ├── __pycache__
│   │   │   ├── model.cpython-39.pyc
│   │   │   └── xgrid_utils.cpython-39.pyc
│   │   ├── run_tests.ipynb
│   │   └── xgrid_utils.py
│   ├── ML_Models
│   │   └── ML_models.ipynb
│   └── Replication
│       ├── d-LSTM_Replication.ipynb
│       ├── EA-LSTM_Enhanced.ipynb
│       ├── EA-LSTM_Enhanced_Separated.ipynb
│       ├── EA-LSTM_Replication.ipynb
│       ├── E-LSTM_Replication.ipynb
│       └── m-LSTM_Replication.ipynb
└── Results
    ├── Improvements
    │   ├── folders containing results of different models
    │   ├── all_results.csv
    │   └── sensitivity_analysis.csv
    ├── ML_Models
    │   ├── csv files containing scores of ML models
    └── Replication
        ├── d-lstm_scores.csv
        ├── ea_lstm_scores.csv
        ├── e_lstm_scores.csv
        └── m-lstm_scores.csv
```
