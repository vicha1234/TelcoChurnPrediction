import os
import sys
import joblib
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import logger

from src.preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train_decision_tree(X_train, y_train, X_test, y_test):
    print("\n--- Training Decision Tree ---")
    logger.info("Starting Decision Tree training")

    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy'],
    }

    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    predictions = best_dt.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    logger.info(f"Decision Tree - Accuracy: {accuracy_score(y_test, predictions):.4f}, F1: {f1_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    return best_dt


def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n--- Training Neural Network ---")
    logger.info("Starting Neural Network training")

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    return model


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess()

    dt_model = train_decision_tree(X_train, y_train, X_test, y_test)
    nn_model = train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(dt_model, "models/decision_tree.pkl")
    nn_model.save("models/neural_network.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n✅ All models saved to models/ folder.")
    logger.info("All models saved successfully")