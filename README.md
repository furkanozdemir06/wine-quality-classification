# 🍷 Wine Quality Prediction

Predicting wine quality scores from chemical properties using a Random Forest classifier. Quality control is a critical step in wine production, and machine learning can help estimate quality from measurable chemical components instead of relying only on expert tasting.

## Highlights

- Multiclass classification on 2,056 labeled wines with 11 chemical features
- Exploratory analysis of the quality distribution, alcohol content, and feature correlations
- Random Forest classifier (800 trees) reaching about **56.6% validation accuracy**
- Ready-to-use `submission.csv` with predictions for the 1,372 test wines

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 2,056 | Wines with chemical measurements and the target `quality` |
| `test.csv` | 1,372 | Wines with chemical measurements only, to be predicted |

Features (all numeric):

| Feature | Feature |
|---------|---------|
| `fixed acidity` | `free sulfur dioxide` |
| `volatile acidity` | `total sulfur dioxide` |
| `citric acid` | `density` |
| `residual sugar` | `pH` |
| `chlorides` | `sulphates` |
| `alcohol` | |

The target `quality` is an integer score, so this is a multiclass classification problem.

## Approach

1. **Exploratory data analysis**
   - Plotted the distribution of quality scores.
   - Compared alcohol content across quality levels with a boxplot.
   - Inspected feature relationships with a correlation heatmap.
2. **Modeling**
   - Split the labeled data 80/20 into training and validation sets.
   - Trained a `RandomForestClassifier` (`n_estimators=800`, `max_depth=10`, `random_state=42`).
3. **Evaluation**
   - Measured accuracy on the validation set.
4. **Prediction**
   - Predicted quality for the test wines and exported the results to `submission.csv`.

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | 0.5655 |

Predicting an exact quality score from chemistry alone is a hard task, and the model learns meaningful patterns but leaves room for improvement. Class imbalance, where some quality levels have very few examples, likely limits performance. Alcohol, sulphates, and volatile acidity appear to be among the most informative features.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn matplotlib seaborn
```

Place `train.csv` and `test.csv` in the project root, then run:

```bash
jupyter notebook WinePrediction.ipynb
```

The notebook generates `submission.csv` with the predicted `quality` for each test wine.

## Project Structure

```
.
├── WinePrediction.ipynb   # EDA, modeling, and prediction
├── train.csv              # Training data (not included)
├── test.csv               # Test data (not included)
├── submission.csv         # Generated predictions
└── README.md
```

## Future Work

- Plot feature importances to confirm which chemical properties drive the predictions.
- Address class imbalance with class weights or resampling.
- Compare other models (Gradient Boosting, XGBoost, LightGBM) and tune hyperparameters with cross-validation.
- Evaluate with per-class metrics such as precision, recall, and F1-score, and a confusion matrix.
- Try framing the problem as regression or ordinal classification, since quality scores are ordered.
- Exclude the `Id` column from the model features.

## Author

Furkan
