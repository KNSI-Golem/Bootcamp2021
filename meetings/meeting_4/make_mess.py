import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

full_data = pd.read_csv('raw_data/cardio_train.csv', delimiter=";").drop("id", axis=1)

full_data['gender'] = full_data.gender.apply(lambda value: "female" if value == 1 else "male")


### Lifestyle feature

def create_lifestyle(array: list):

    lifestyle_string_comp = []
    if array[0] != 0:
        lifestyle_string_comp.append("alcoholic")
    if array[1] != 0:
        lifestyle_string_comp.append("smoker")
    if array[2] != 0:
        lifestyle_string_comp.append("active")

    if len(lifestyle_string_comp) == 0:
        return np.NaN

    return ', '.join(lifestyle_string_comp)


def create_cat_med_feature(value: int):
    val_dict = {
        1: "normal",
        2: "above normal",
        3: "way above normal"
    }

    return val_dict[value]

full_data['lifestyle'] = full_data[['alco', 'smoke', 'active']].apply(create_lifestyle, axis=1)
full_data.drop(['alco', 'smoke', 'active'], axis=1, inplace=True)

full_data['cholesterol_lvl'] = full_data['cholesterol'].apply(create_cat_med_feature)
full_data['glucose_lvl'] = full_data['gluc'].apply(create_cat_med_feature)

full_data.drop(['cholesterol', 'gluc'], axis=1, inplace=True)

full_data['is_cardio_ill'] = full_data['cardio']
full_data.drop(['cardio'], axis=1, inplace=True)

### Missing values

age_null_indices = sorted(np.random.randint(0, len(full_data), len(full_data) * 3 // 100))
height_null_values = np.random.randint(0, len(full_data), len(full_data) * 2 // 100)
weight_null_values = np.random.randint(0, len(full_data), len(full_data) * 1 // 100)

full_data.loc[age_null_indices, 'age'] = np.nan
full_data.loc[height_null_values, 'height'] = np.nan
full_data.loc[weight_null_values, 'weight'] = np.nan

train_X, test_X, train_y, test_y = train_test_split(
    full_data.drop("is_cardio_ill", axis=1),
    full_data['is_cardio_ill'],
    test_size=0.7,
    random_state=42
)



train_data = pd.concat([train_X, train_y], axis=1)

if not "data" in os.listdir(os.getcwd()):
    os.mkdir(f"{os.getcwd()}{os.sep}data")

train_data.to_csv(f'{os.getcwd()}{os.sep}data{os.sep}train_data.csv', index=False)
test_X.to_csv(f'{os.getcwd()}{os.sep}data{os.sep}test_features.csv')
test_y.to_csv(f'{os.getcwd()}{os.sep}data{os.sep}test_labels.csv')
