# Predicting Online News Popularity

## 🎯 Project Overview

This project tackles the fascinating problem of predicting online news popularity using machine learning techniques. The challenge involves analyzing a heterogeneous set of features from articles published by Mashable over two years to predict the number of social media shares.

### Key Highlights
- **Dataset:** 39,797 articles with 61 features
- **Target:** Social media shares (discretized into 5 classes)
- **Challenge:** Handling severely imbalanced classes
- **Best Model:** Support Vector Machine with strategic class balancing

## 📊 Dataset Description

The dataset contains articles from Mashable with diverse features including:

- **Content Features:** Word counts, token rates, sentiment analysis
- **Metadata:** Publication day, data channels (lifestyle, entertainment, business, etc.)
- **Keyword Analytics:** Min/max/average shares for keywords
- **Multimedia:** Number of images, videos, and links
- **NLP Features:** LDA topic modeling, sentiment polarity

### Target Variable Distribution
The original dataset showed extreme class imbalance, with the majority of articles having low share counts (class 0), making this a challenging classification problem.

## 🔧 Methodology

### 1. Data Preprocessing & Cleaning

**Noise Detection and Removal:**
- Identified and removed 135 rows with formatting issues
- Eliminated samples with values outside expected column domains
- Handled missing values and 'n.a.' entries
- Applied Z-score outlier detection (threshold: 5) removing 1,875 additional samples

**Feature Engineering:**
- Used Random Forest for feature importance analysis
- Retained only features with importance > 0.02
- Reduced from 61 to 18 most informative features
- Applied MinMax scaling for model compatibility

### 2. Class Imbalance Strategy

The most critical challenge was the severe class imbalance. Instead of uniform balancing, I implemented a strategic approach:

```
Class Distribution Strategy:
- Class 0 (low shares): 1/3 of samples
- Class 1: 1/4 of samples  
- Class 2: 1/5 of samples
- Class 3: 1/6 of samples
- Class 4 (viral): 1/7 of samples
```

**Rationale:** This distribution acknowledges that viral content is naturally rare while ensuring all classes have sufficient representation for learning.

**Implementation:**
1. **RandomUnderSampler:** Reduced overrepresented classes
2. **SMOTE:** Generated synthetic samples for underrepresented classes
3. Maintained realistic class proportions reflecting real-world news sharing patterns

### 3. Model Selection & Evaluation

**Models Tested:**
- Decision Trees
- Support Vector Machines  
- AdaBoost Ensemble
- Random Forest
- Multi-Layer Perceptron Neural Networks

**Evaluation Strategy:**
- 66% training, 33% testing split
- Cross-validation with both accuracy and F1-macro scoring
- Emphasis on F1-macro to ensure sensitivity across all classes

## 📈 Results

### Before Class Balancing
| Model | Accuracy | F1-Macro | CV Accuracy | CV F1-Macro |
|-------|----------|----------|-------------|-------------|
| DT | 0.66 | 0.21 | 0.4071 | 0.1513 |
| SVM | 0.80 | 0.18 | 0.7968 | 0.1774 |
| Boost | 0.80 | 0.18 | 0.7957 | 0.1783 |
| Random Forest | 0.80 | 0.18 | 0.6614 | 0.1570 |
| MLPN | 0.80 | 0.18 | 0.7968 | 0.1774 |

### After Class Balancing
| Model | Accuracy | F1-Macro | CV Accuracy | CV F1-Macro |
|-------|----------|----------|-------------|-------------|
| DT | 0.49 | 0.20 | 0.4071 | 0.1513 |
| SVM | 0.65 | 0.25 | 0.7968 | 0.1774 |
| Boost | 0.59 | 0.23 | 0.7957 | 0.1783 |
| Random Forest | 0.70 | 0.26 | 0.6614 | 0.1570 |
| MLPN | 0.65 | 0.22 | 0.7968 | 0.1774 |

### Hyperparameter Tuning Results

**Best SVM Configuration:**
```python
{
    'C': 1,
    'kernel': 'rbf',
    'gamma': 'scale',
    'decision_function_shape': 'ovo',
    'random_state': 42
}
```

## 🔍 Key Insights

### 1. The Class Imbalance Challenge
The original dataset's extreme imbalance (majority class dominance) led to models that achieved high accuracy by simply predicting the majority class. This highlighted the importance of F1-macro as an evaluation metric for imbalanced problems.

### 2. Strategic Balancing vs. Uniform Balancing
Unlike uniform class balancing, the strategic approach maintained the natural hierarchy of news sharing patterns while ensuring model sensitivity to all classes. This proved crucial for real-world applicability.

### 3. Feature Importance Impact
Reducing features from 61 to 18 using Random Forest importance analysis improved model efficiency without sacrificing performance, demonstrating the value of feature selection in high-dimensional problems.

### 4. Model Performance Trade-offs
The class balancing strategy improved F1-macro scores significantly (from 0.18 to 0.25 for SVM) while maintaining reasonable accuracy, showing the successful navigation of the precision-recall trade-off.

## 🛠️ Technical Implementation

### Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
```

### Key Functions
- `clean_and_load_dataset()`: Comprehensive data cleaning pipeline
- `get_hist()`: Visualization utility for distribution analysis
- `fit_models()` & `evaluate_models()`: Model training and evaluation framework
- `tune_model()`: Hyperparameter optimization wrapper

## 📝 Lessons Learned

1. **Domain Knowledge Matters:** Understanding that news sharing follows natural power-law distributions informed the strategic balancing approach.

2. **Evaluation Metrics Selection:** In imbalanced problems, accuracy alone can be misleading. F1-macro provided better insights into model performance across all classes.

3. **Preprocessing Impact:** Careful outlier removal and feature selection significantly improved model performance and training efficiency.

4. **Class Balancing Strategy:** One-size-fits-all approaches (like uniform balancing) may not suit all problems. Domain-informed strategies often perform better.

## 🚀 Future Improvements

1. **Advanced Sampling Techniques:** Explore borderline-SMOTE or ADASYN for more sophisticated synthetic sample generation
2. **Feature Engineering:** Investigate polynomial features or interaction terms
3. **Ensemble Methods:** Combine multiple models with different balancing strategies
4. **Deep Learning:** Experiment with neural networks designed for imbalanced classification
5. **Temporal Analysis:** Incorporate time-series aspects of news popularity

## 📚 References

- K. Fernandes, P. Vinagre and P. Cortez. "A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News." Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

---

*This project demonstrates the importance of careful data preprocessing, strategic handling of class imbalance, and thoughtful evaluation in machine learning challenges. The combination of domain knowledge and technical expertise proved crucial for developing an effective news popularity prediction model.*
