# EV Charging Demand & Cost Predictor ⚡

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red.svg)

## Overview
A machine learning architecture designed to predict Electric Vehicle (EV) charging station demand(energy) and dynamic cost fluctuations. By analyzing historical charging metrics and temporal data, this system allows operators to optimize grid load and pricing strategies in real-time.

## System Architecture
* **Predictive Engine:** LightGBM (Gradient Boosting Framework) optimized for high-speed tabular data inference and low memory consumption.
* **Data Processing:** Robust feature engineering, handling missing values, and temporal data manipulation using standard Python data structures and NumPy.
* **Interactive Frontend:** Deployed via Streamlit to allow end-users to input live variables and retrieve instant cost/demand predictions.

## Key Features
* **High-Speed Inference:** Utilizes LightGBM's leaf-wise tree growth algorithm for faster training and prediction compared to standard XGBoost/Random Forest models.
* **Dynamic Cost Modeling:** Factors in temporal constraints (peak hours, weekend shifts) to predict pricing surges.
* **Web Deployment:** A clean, container-ready Streamlit interface for non-technical stakeholders to test the model logic.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ev-demand-predictor.git](https://github.com/yourusername/ev-demand-predictor.git)
   cd ev-demand-predictor
