
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def cargar_datos(path):
    """Carga el dataset desde el archivo CSV."""
    return pd.read_csv(path)

def preprocesar_datos(df):
    """Aplica codificación y limpieza al DataFrame."""
    df = df.copy()
    df['Riesgo Vital'] = df['Riesgo Vital'].map({'Sí': 1, 'No': 0})
    df['Intervención Urgente'] = df['Intervención Urgente'].map({'Sí': 1, 'No': 0})
    le = LabelEncoder()
    df['Motivo'] = le.fit_transform(df['Motivo'])
    return df

def dividir_datos(df, target='ESI', test_size=0.2, random_state=42):
    """Separa los datos en entrenamiento y prueba."""
    X = df.drop(columns=target)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def entrenar_modelo(X_train, y_train):
    """Entrena un modelo Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluar_modelo(model, X_test, y_test):
    """Evalúa el modelo y retorna el reporte de clasificación."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def guardar_modelo(model, path):
    """Guarda el modelo entrenado en disco."""
    joblib.dump(model, path)
