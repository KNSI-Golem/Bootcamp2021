import pandas as pd
import numpy as np


def make_hm_classification(train_X: pd.DataFrame,
                        train_y: pd.Series,
                        test_X: pd.DataFrame):
    
    model = LogisticRegression(random_state=42)
    model.fit(train_X, test_X)
    
    predictions = model.predict(test_X)
    ground_truth = pd.read_csv('data/test_set.csv')['is_cardio_ill']

    acc = accuracy_score(ground_truth, predictions)

    return acc
