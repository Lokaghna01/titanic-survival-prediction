import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore

def preprocess_data(df):
    df = df.copy()

    # Drop columns with too much missing data or not useful
    df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

    # Fill missing values
    df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)


    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders, scaler
