# Sea Ice Extent Forecasting using Convolutional LSTMs
<!-- Term project for SYDE 675: Sea ice extent forecasting using LSTMs, attention, and multiple timeframes -->

## Abstract
In this paper, a data-driven approach is taken to predict Arctic sea ice extent (SIE). We first assess the work of (Ali et al., 2022) in their paper ”Sea Ice Forecasting using Attention-based Ensemble LSTM”. We find that their results are not perfectly reproducible, and that their model does not outperform a naive statistical baseline. We then propose an alternative model architecture which accepts both spatial and temporal inputs, and uses several new climatic input variables. We test a variety of hyperparameters and find our best model outperforming a baseline statistical model by 56% in %RMSE.

## Folder Structure
.
+-- Data
+-- Models
|   +-- Improvements
|   +-- ML_Models
|   +-- Replication
+-- Results
|   +-- Improvements
|   +-- ML_Models
|   +-- Replication
