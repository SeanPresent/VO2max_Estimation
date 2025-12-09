# VO₂max Estimation using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

Machine learning-based estimation of continuous oxygen uptake (VO₂) and maximal oxygen consumption (VO₂max) for real-time fitness assessment and wearable device applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements a machine learning regression model to estimate continuous oxygen uptake (VO₂) and maximal oxygen consumption (VO₂max) using physiological parameters. The model is designed for real-time applications in wearable devices, enabling personalized fitness tracking, training optimization, and health risk assessment.

**Key Contribution**: Development of a robust XGBoost-based regression model that achieves exceptional performance metrics for continuous VO₂ estimation, suitable for integration into wearable fitness devices.

## ✨ Features

- **Real-time VO₂ Estimation**: Continuous monitoring of oxygen consumption during exercise
- **VO₂max Prediction**: Estimation of maximal oxygen consumption capacity
- **Multiple ML Models**: Comparison of various regression algorithms (XGBoost, Random Forest, etc.)
- **Feature Engineering**: Comprehensive preprocessing including anomaly detection and data validation
- **VO₂max Categorization**: Classification based on ACSM fitness standards (Poor, Fair, Good, Excellent, Superior)
- **Cross-validation**: Robust model evaluation using k-fold cross-validation

## 📊 Dataset

The dataset is sourced from **PhysioNet** and contains cardiorespiratory measurements from 992 treadmill tests conducted at the University of Malaga's Exercise Physiology and Human Performance Lab (2008-2018).

### Dataset Characteristics
- **Participants**: 992 individuals
- **Age Range**: 10-63 years
- **Population**: Amateur and professional athletes
- **Monitoring**: Breath-by-breath physiological parameter tracking

### Features Used
- `Age`: Participant age (years)
- `Weight`: Body weight (kg)
- `Height`: Body height (cm)
- `HR`: Heart rate (bpm)
- `Sex`: Gender (0: Male, 1: Female)
- `Time`: Exercise time (minutes)
- `VO2`: Oxygen consumption (mL/min) → converted to `VO2_ml_kg_min` (mL/kg/min)

## 🔬 Methodology

### Data Preprocessing
1. **Data Merging**: Combined subject information and test measurements
2. **Unit Conversion**: Converted VO₂ from mL/min to mL/kg/min
3. **Anomaly Detection**: Filtered outliers using ACSM fitness category criteria
4. **Age Filtering**: Excluded participants outside 19-80 years range
5. **Data Validation**: Removed invalid VO₂max categories

### Model Development
1. **Feature Selection**: Selected key physiological parameters (Age, Weight, Height, HR, Sex, Time)
2. **Train-Test Split**: Subject-level split (80% train, 20% test) to prevent data leakage
3. **Model Comparison**: Evaluated multiple regression models using PyCaret
4. **Hyperparameter Tuning**: Optimized XGBoost using MAPE (Mean Absolute Percentage Error)
5. **Cross-Validation**: 5-fold cross-validation for robust evaluation

### VO₂max Categorization
The model categorizes VO₂max values based on ACSM standards:
- **Superior**: Highest fitness level
- **Excellent**: High fitness level
- **Good**: Above average fitness
- **Fair**: Average fitness
- **Poor**: Below average fitness

## 📈 Model Performance

The optimized XGBoost model achieved the following performance metrics on the test dataset (132 participants):

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | 0.1793 |
| **MSE** (Mean Squared Error) | 0.1460 |
| **RMSE** (Root Mean Squared Error) | 0.3821 |
| **R²** (R-squared) | 0.9991 |
| **RMSLE** (Root Mean Squared Logarithmic Error) | 0.0140 |
| **MAPE** (Mean Absolute Percentage Error) | 0.0066 |

### Performance Highlights
- **R² = 0.9991**: Exceptional model fit, explaining 99.91% of variance
- **MAPE = 0.0066**: Less than 1% average percentage error
- **Low RMSE**: Minimal prediction error in VO₂ estimation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

자세한 설치 가이드는 [QUICK_START.md](QUICK_START.md)를 참조하세요.

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/VO2max_Estimation.git
cd VO2max_Estimation
```

2. **Create project structure** (optional, for organized structure)
```bash
bash setup_project.sh
```

3. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
- pandas
- numpy
- scikit-learn
- xgboost
- pycaret
- matplotlib
- seaborn
- torch (optional, for GPU acceleration)

## 💻 Usage

### Basic Usage

1. **Prepare your data**
   - Ensure you have `subject-info.csv` and `test_measure.csv` files
   - Place them in the `data/` directory

2. **Run the main script**
```bash
python xgboost_ml.py
```

### Model Training Workflow

The script performs the following steps:
1. Load and merge datasets
2. Data preprocessing and feature engineering
3. Anomaly detection and filtering
4. Train-test split (subject-level)
5. Model comparison and selection
6. XGBoost hyperparameter tuning
7. Model evaluation and visualization
8. Save the trained model

### Using the Trained Model

```python
from pycaret.regression import load_model, predict_model
import pandas as pd

# Load the saved model
model = load_model('models/20240423_best_tuned_xgb_VO2max_model')

# Prepare new data
new_data = pd.DataFrame({
    'Age': [30],
    'Weight': [70],
    'Height': [175],
    'HR': [150],
    'Sex': [0],
    'time': [10]
})

# Make predictions
predictions = predict_model(model, data=new_data)
print(predictions['Label'])  # Predicted VO2_ml_kg_min
```

## 📁 Project Structure

```
VO2max_Estimation/
│
├── README.md                 # Project documentation
├── PROJECT_STRUCTURE.md      # Detailed project structure guide
├── QUICK_START.md           # Quick start guide
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── README.md           # Data description
│
├── src/                     # Source code modules
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering modules
│   ├── models/              # Model training and evaluation
│   ├── visualization/       # Plotting functions
│   └── utils/               # Utility functions
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── scripts/                 # Executable scripts
│   └── train_model.py       # Model training script
│
├── models/                  # Saved models
│   └── best_xgboost_model.pkl
│
├── results/                 # Results and outputs
│   ├── figures/            # Generated plots
│   └── reports/             # Evaluation reports
│
├── config/                  # Configuration files
│   └── config.yaml.example  # Example configuration
│
├── tests/                   # Unit tests
│   └── test_preprocessing.py
│
└── docs/                    # Additional documentation
    └── methodology.md
```

> **📌 프로젝트 구조 개선**: 현재 프로젝트를 더 전문적으로 만들기 위한 구조 개선 방안은 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)를 참조하세요.

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{hong2024machine,
  title={Machine Learning Regressors to Estimate Continuous Oxygen Uptakes ($\dot{V}O_2$)},
  author={Hong, Daeeon and Sun, Sukkyu},
  journal={Applied Sciences},
  volume={14},
  number={17},
  pages={7888},
  year={2024},
  publisher={MDPI}
}
```

**Paper**: Hong, D.; Sun, S. Machine Learning Regressors to Estimate Continuous Oxygen Uptakes (V̇O₂). *Appl. Sci.* 2024, 14, 7888. https://doi.org/10.3390/app14177888

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

See the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Daeeon Hong** - Department of Interdisciplinary Program in Bioengineering, Seoul National University
- **Sukkyu Sun** - Department of AI Software Convergence, Dongguk University (Corresponding Author)

## 🙏 Acknowledgments

- University of Malaga's Exercise Physiology and Human Performance Lab for providing the dataset
- PhysioNet for hosting the dataset
- PyCaret community for the excellent ML framework

## 📧 Contact

For questions or collaborations, please contact:
- **Sukkyu Sun**: sukkyu.sun@dgu.ac.kr
- **Daeeon Hong**: shong65@snu.ac.kr

---

**Note**: This project is part of research published in Applied Sciences (MDPI). For detailed methodology and results, please refer to the published paper.

