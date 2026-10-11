
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("student_data.csv")

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_marks",
    "sleep_hours"
]
target = "final_score"

X = df[features]
y = df[target]

# 2. Split data before fitting the scaler
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Build the Artificial Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# 6. Train the network
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate on unseen test data
predictions = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.3f}")

# 8. Save the model and preprocessing scaler
model.save("student_ann.keras")

import joblib
joblib.dump(scaler, "scaler.pkl")

# 9. Plot training and validation loss
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("ANN Training Performance")
plt.legend()
plt.tight_layout()
plt.show()