
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def load_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Select the relevant columns from each dataset
    df1 = df1[['v1', 'v2']].copy()
    df2 = df2[['label', 'text']].copy()

    # Give both datasets consistent column names
    df1.columns = ['label', 'text']

    df = pd.concat([df1, df2], ignore_index=True)

    print("Original rows:", len(df))
    print("Duplicates:", df.duplicated(subset='text').sum())

    conflicts = df.groupby('text')['label'].nunique()
    conflicts = conflicts[conflicts > 1]

    if len(conflicts):
        raise ValueError(
            f"{len(conflicts)} duplicate emails have conflicting labels."
        )

    df = df.drop_duplicates(subset='text').reset_index(drop=True)

    return df


def get_models():
    return {
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english'
            )),
            ('model', MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english'
            )),
            ('model', LogisticRegression(max_iter=1000))
        ]),

        "SVC": Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english'
            )),
            ('model', SVC())
        ])
    }


def evaluate_models(X, y):
    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    predictions = {}

    for name, model in models.items():

        oof = np.empty(len(y), dtype=object)
        fold_metrics = []

        for train_idx, val_idx in cv.split(X, y):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            oof[val_idx] = pred

            tn, fp, fn, tp = confusion_matrix(
                y_val, pred, labels=['ham', 'spam']
            ).ravel()

            fold_metrics.append({
                'Accuracy': accuracy_score(y_val, pred),
                'Macro-F1': f1_score(y_val, pred, average='macro'),
                'Spam Precision': precision_score(
                    y_val, pred, pos_label='spam', zero_division=0
                ),
                'Spam Recall': recall_score(
                    y_val, pred, pos_label='spam', zero_division=0
                ),
                'Spam F1': f1_score(
                    y_val, pred, pos_label='spam', zero_division=0
                ),
                'Ham FPR': fp / (fp + tn) if (fp + tn) else 0
            })

        fold_df = pd.DataFrame(fold_metrics)

        results.append({
            'Model': name,
            **{
                metric: f"{fold_df[metric].mean():.4f} ± {fold_df[metric].std():.4f}"
                for metric in fold_df.columns
            }
        })

        predictions[name] = pd.Series(oof, index=X.index)

    return pd.DataFrame(results), predictions


def analyze_errors(X, y, predictions):
    errors = pd.DataFrame({
        'Email': X,
        'Actual': y,
        'Predicted': predictions
    })

    fp = errors[
        (errors['Actual'] == 'ham') &
        (errors['Predicted'] == 'spam')
    ]

    fn = errors[
        (errors['Actual'] == 'spam') &
        (errors['Predicted'] == 'ham')
    ]

    return fp, fn


def train_final_model(X, y, model_name='SVC'):
    model = get_models()[model_name]
    model.fit(X, y)
    return model


def predict_test(model, test_file, text_column='text'):
    test_df = pd.read_csv(test_file)

    predictions = model.predict(test_df[text_column])

    return test_df, predictions


def export_predictions(predictions, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(predictions))
