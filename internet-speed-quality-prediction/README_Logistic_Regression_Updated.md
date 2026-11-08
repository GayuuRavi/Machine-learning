# Internet Connection Quality Prediction --- Logistic Regression

## Project overview

This project uses Logistic Regression to classify an internet connection
as `Good` or `Poor` based on network and connection characteristics.

The practice dataset contains 1,200 simulated records. It is intended
for learning and is not real customer or production measurement data.

## Objective

Predict `connection_quality` using: - `latency_ms` - `packet_loss_pct` -
`signal_strength_pct` - `subscribed_bandwidth_mbps` -
`connected_devices` - `wifi_band` - `time_of_day`

The target is `connection_quality`.

**Target leakage warning:** Do not include `download_speed_mbps` as an
input feature because the `connection_quality` label is derived from
download speed.

## Notebook workflow

1.  Load and inspect the dataset.
2.  Encode the target labels.
3.  Encode categorical input features.
4.  Define `X` (input features) and `Y` (target).
5.  Split into training and testing datasets.
6.  Fit Logistic Regression using training data.
7.  Generate class predictions and probabilities.
8.  Evaluate using accuracy, precision, recall, F1-score, confusion
    matrix, ROC curve, and ROC-AUC.

## Current implementation

The latest notebook correctly fits the model using:

``` python
logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(x_train, y_train)
```

It then generates predictions for both training and test data.

## Results currently recorded in the notebook

  Metric       Training   Testing
  ---------- ---------- ---------
  Accuracy       94.79%    95.00%
  ROC-AUC        0.9862    0.9860

Test classification report:

  Class        Precision   Recall   F1-score   Support
  ---------- ----------- -------- ---------- ---------
  Good (0)          0.98     0.95       0.96       170
  Poor (1)          0.89     0.94       0.92        70

Test confusion matrix:

``` text
[[162   8]
 [  4  66]]
```

The class order is:

``` python
le_encoder.classes_
# ['Good', 'Poor']
```

Therefore, the confusion matrix rows and columns are ordered as Good,
then Poor. The model correctly identifies 66 of 70 Poor connections and
misses 4.

**Interpretation:** The recorded training and test metrics are close,
which is encouraging and shows no obvious train/test performance gap.
The test ROC-AUC of approximately 0.986 indicates strong class
separation on this split. These results are from one split of simulated
data and should not be treated as production performance.

## Recommended improvements before finalizing

### 1. Stratify the split

The notebook currently uses `stratify=None`. For this binary
classification task, use:

``` python
x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=34,
    shuffle=True,
    stratify=Y
)
```

This helps preserve the Good/Poor class proportions in both subsets.

### 2. Avoid redundant dummy variables

The notebook currently retains both Wi-Fi dummy columns and all four
time-of-day dummy columns. With an intercept, this creates redundant
columns.

Use:

``` python
internet_speed = pd.get_dummies(
    internet_speed,
    columns=["wifi_band", "time_of_day"],
    drop_first=True,
    dtype=int
)
```

Then build `X` from the encoded predictors, excluding
`download_speed_mbps` and `connection_quality`.

### 3. Fix the LabelEncoder warning

Use a one-dimensional Series:

``` python
internet_speed["connection_quality"] = le_encoder.fit_transform(
    internet_speed["connection_quality"]
)
```

### 4. Scale features within a Pipeline

The predictors have different numerical ranges. Scaling is recommended
for Logistic Regression. A Pipeline ensures the scaler is fitted on
training data only:

``` python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

logistic_model.fit(x_train, y_train)
```

### 5. Calculate ROC-AUC from probabilities

Use predicted probabilities, not hard predicted labels:

``` python
y_test_proba = logistic_model.predict_proba(x_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
```

The notebook correctly does this in its final test ROC/AUC cell. Since
the encoder class order is `['Good', 'Poor']`, probability column
`[:, 1]` represents the Poor class.

## Evaluation metrics explained

-   **Accuracy:** Proportion of all records classified correctly.
-   **Precision:** Of the records predicted as a class, how many
    actually belong to that class?
-   **Recall:** Of all actual records in a class, how many did the model
    identify?
-   **F1-score:** Harmonic mean of precision and recall.
-   **Confusion matrix:** Counts correct and incorrect predictions by
    actual and predicted class.
-   **ROC curve:** Plots true positive rate against false positive rate
    across thresholds.
-   **ROC-AUC:** Summarizes the model's ability to rank positive-class
    records above negative-class records across thresholds.

For this use case, pay particular attention to recall for the `Poor`
class because false negatives mean genuinely Poor connections are
classified as Good.

## Next steps

1.  Apply stratification and remove redundant dummy columns.
2.  Add scaling using a Pipeline.
3.  Re-run the notebook and record the updated metrics.
4.  Compare Logistic Regression with Decision Tree and Random Forest
    classifiers using the same train/test split.
5.  Compare test precision, recall, F1-score, ROC-AUC, and confusion
    matrices---not accuracy alone.
