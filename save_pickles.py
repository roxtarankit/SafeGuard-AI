import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_kiva_rf_model(df, output_model='kiva_model.pkl', output_scaler='kiva_scaler.pkl', output_features='kiva_features.pkl'):
    # Drop unnecessary columns
    df = df.drop(['activity', 'id', 'use', 'tags', 'country', 'date', 'region', 'partner_id'], axis=1)

    # Convert datetime columns
    df['posted_time'] = pd.to_datetime(df['posted_time'])
    df['posted_time_y'] = df['posted_time'].dt.year
    df['posted_time_m'] = df['posted_time'].dt.month
    df['posted_time_d'] = df['posted_time'].dt.day
    df = df.drop('posted_time', axis=1)

    for col in ['disbursed_time', 'funded_time']:
        df[col] = pd.to_datetime(df[col].fillna(df[col].mode()[0]))
        df[col + '_y'] = df[col].dt.year
        df[col + '_m'] = df[col].dt.month
        df[col + '_d'] = df[col].dt.day
        df = df.drop(col, axis=1)

    # Gender count features
    def male_count(x):
        return str(x).split(', ').count('male')

    def female_count(x):
        return str(x).split(', ').count('female')

    df['male_count'] = df['borrower_genders'].apply(male_count)
    df['female_count'] = df['borrower_genders'].apply(female_count)
    df = df.drop('borrower_genders', axis=1)

    # Target and features
    y = df['repayment_interval'].replace({'irregular': 0, 'bullet': 1, 'monthly': 3})
    X = df.drop('repayment_interval', axis=1)

    # One-hot encode object columns
    X = pd.get_dummies(X)

    # Save feature columns for later use in prediction
    feature_columns = X.columns.tolist()

    # Split and scale
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train Random Forest
    model = RandomForestClassifier()
    model.fit(x_train_scaled, y_train)

    # Evaluate (optional)
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")

    # Save the model, scaler, and features
    joblib.dump(model, output_model)
    joblib.dump(scaler, output_scaler)
    joblib.dump(feature_columns, output_features)

    print(f"Saved model to {output_model}")
    print(f"Saved scaler to {output_scaler}")
    print(f"Saved features to {output_features}")

cols = [
    'id', 'funded_amount', 'loan_amount', 'activity', 'sector', 'use',
    'country', 'region', 'currency', 'partner_id', 'posted_time',
    'disbursed_time', 'funded_time', 'term_in_months', 'lender_count',
    'tags', 'borrower_genders', 'repayment_interval', 'date'
]
df = pd.read_csv('kiva_loans_small23_Final1.csv', names=cols)
df_health = df[df['activity'] == 'Health']  # Optional filtering
train_and_save_kiva_rf_model(df_health)