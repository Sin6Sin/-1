import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def preprocess_transactions(df):
    """Полный пайплайн предобработки транзакций"""
    
    df['Merchant'] = df['Merchant'].fillna('UNKNOWN')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    vectorizer = CountVectorizer()
    merchant_vectors = vectorizer.fit_transform(df['Merchant']).toarray()
    
    numeric_features = ['Transaction_Amount']
    categorical_features = ['Transaction_Type', 'day_of_week', 'month', 'is_weekend']
    
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_features])
    
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_encoded = encoder.fit_transform(df[categorical_features])
    
    feature_matrix = np.hstack([
        numeric_scaled,
        categorical_encoded,
        merchant_vectors
    ])
    
    return feature_matrix, vectorizer, scaler, encoder

if __name__ == "__main__":
    df = pd.read_csv('daily_transactions.csv')
    X, vec, scl, enc = preprocess_transactions(df)
    print(f"Размер итоговой матрицы признаков: {X.shape}")
