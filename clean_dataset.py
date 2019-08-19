import pandas as pd
import joblib

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def main():
    data = pd.read_csv('data/data.csv')
    data = data[pd.notnull(data['Tuition and fees'])]

    min_fields = 18
    left = 0

    for index, row in data.iterrows():
        filled = 0

        for name, field in row.items():
            if str(field) != 'nan':
                filled += 1

        if filled > min_fields:
            left += 1
        else:
            data.drop(index, inplace=True)
    print(left)

    # Split into train and test sets
    data = data.reset_index(drop=True)
    train, test = train_test_split(data, test_size=0.2)
    X_train, y_train = train.drop(train.columns[[2]], axis=1), train.iloc[:, 2]
    X_test, y_test = test.drop(test.columns[[2]], axis=1), test.iloc[:, 2]

    # Scale targets
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(y_train.values.reshape(-1, 1))
    y_train, y_test = scaler.transform(y_train.values.reshape(-1, 1)), scaler.transform(y_test.values.reshape(-1, 1))
    joblib.dump(scaler, 'min_max_scaler.pkl')

    # Encode
    numeric_features = X_train.columns[[0, 1] + [i for i in range(4, 24)]]
    numeric_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                    ('scaler', preprocessing.MinMaxScaler())])

    column_features = X_train.columns[[2, 3]]
    column_trans = Pipeline(steps=[('encoder', preprocessing.OneHotEncoder(drop='first'))])

    transformer = ColumnTransformer(transformers=[('numeric', numeric_trans, numeric_features),
                                                  ('categorical', column_trans, column_features)],
                                    remainder='passthrough')
    transformer.fit(X_train)

    column_names = transformer.named_transformers_['categorical'].named_steps['encoder'].get_feature_names()
    X_train = pd.DataFrame(transformer.transform(X_train), columns=list(numeric_features) + list(column_names))
    X_test = pd.DataFrame(transformer.transform(X_test), columns=list(numeric_features) + list(column_names))

    X_train.to_csv('data/cleaned_data_train_x.csv')
    X_test.to_csv('data/cleaned_data_test_x.csv')

    pd.DataFrame(y_train).to_csv('data/cleaned_data_train_y.csv', header='College tuition')
    pd.DataFrame(y_test).to_csv('data/cleaned_data_test_y.csv', header='College tuition')


if __name__ == '__main__':
    main()
