import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from prepare_data import prepare_dataset
import pydot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and prepare dataset
file_paths = [f'/Users/sudaxin/Desktop/data_preprocessed_python/s{str(i).zfill(2)}.dat' for i in range(1, 33)]
selected_channels = ['Fp1', 'Fp2']
X, y = prepare_dataset(file_paths, selected_channels)

# Ensure no values exceed the range of 1 to 9
y = np.clip(y, 1, 9)

# Normalize the features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize y to [-1, 1]
y = (y - 5) / 4

# Print the distribution of y values
print("Valence range:", np.min(y[:, 0]), np.max(y[:, 0]))
print("Arousal range:", np.min(y[:, 1]), np.max(y[:, 1]))

# Ensure the data has enough samples
print(f"Total samples: {X.shape[0]}")

# Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
valence_reports = []
arousal_reports = []
valence_auc_scores = []
arousal_auc_scores = []
# Initialize lists to store the metrics
mse_scores = []
rmse_scores = []
mae_scores = []
r2_scores = []
classification_accuracy_scores = []

# Define the CNN-Transformer model
def create_transformer_model(input_shape, num_heads=6, ff_dim=128):
    inputs = layers.Input(shape=input_shape)

    # Transformer block
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    ffn_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = layers.Dense(input_shape[-1])(ffn_output)
    x = layers.Add()([inputs, ffn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Flatten and add dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = layers.Dense(2, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

input_shape = (X.shape[1], 1)  # 例として入力形状を指定
model = create_transformer_model(input_shape)
plot_model(model, to_file='/Users/sudaxin/Desktop/EmoEst transformer result/model_plot.png', show_shapes=True, show_layer_names=True)

fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Ensure the data is correctly reshaped
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Initialize model for each fold to avoid state carryover
    model = create_transformer_model((X.shape[1], 1))

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    predictions = model.predict(X_test)
    # Calculate MSE, RMSE, MAE, R2 for Valence and Arousal
    mse_valence = mean_squared_error(y_test[:, 0], predictions[:, 0])
    mse_arousal = mean_squared_error(y_test[:, 1], predictions[:, 1])
    mse_scores.append((mse_valence, mse_arousal))

    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    rmse_scores.append((rmse_valence, rmse_arousal))

    mae_valence = mean_absolute_error(y_test[:, 0], predictions[:, 0])
    mae_arousal = mean_absolute_error(y_test[:, 1], predictions[:, 1])
    mae_scores.append((mae_valence, mae_arousal))

    r2_valence = r2_score(y_test[:, 0], predictions[:, 0])
    r2_arousal = r2_score(y_test[:, 1], predictions[:, 1])
    r2_scores.append((r2_valence, r2_arousal))

    # Calculate classification accuracy for High/Low classification
    valence_true = y_test[:, 0] > 0
    valence_pred = predictions[:, 0] > 0
    arousal_true = y_test[:, 1] > 0
    arousal_pred = predictions[:, 1] > 0

    valence_accuracy = np.mean(valence_true == valence_pred)
    arousal_accuracy = np.mean(arousal_true == arousal_pred)
    classification_accuracy_scores.append((valence_accuracy, arousal_accuracy))

    valence_report = classification_report(valence_true, valence_pred, output_dict=True)
    valence_reports.append(valence_report)
    valence_report_df = pd.DataFrame(valence_report).transpose()
    valence_report_df.to_csv(
        f'/Users/sudaxin/Desktop/EmoEst transformer result/valence_classification_report_fold_{fold}.csv')

    arousal_report = classification_report(arousal_true, arousal_pred, output_dict=True)
    arousal_reports.append(arousal_report)
    arousal_report_df = pd.DataFrame(arousal_report).transpose()
    arousal_report_df.to_csv(
        f'/Users/sudaxin/Desktop/EmoEst transformer result/arousal_classification_report_fold_{fold}.csv')

    valence_conf_matrix = confusion_matrix(valence_true, valence_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(valence_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Valence Confusion Matrix Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'/Users/sudaxin/Desktop/EmoEst transformer result/valence_confusion_matrix_fold_{fold}.png')
    plt.close()

    arousal_conf_matrix = confusion_matrix(arousal_true, arousal_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(arousal_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Arousal Confusion Matrix Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'/Users/sudaxin/Desktop/EmoEst transformer result/arousal_confusion_matrix_fold_{fold}.png')
    plt.close()

    valence_fpr, valence_tpr, _ = roc_curve(valence_true, predictions[:, 0])
    valence_roc_auc = auc(valence_fpr, valence_tpr)
    valence_auc_scores.append(valence_roc_auc)
    plt.figure()
    plt.plot(valence_fpr, valence_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % valence_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Valence Fold {fold}')
    plt.legend(loc="lower right")
    plt.savefig(f'/Users/sudaxin/Desktop/EmoEst transformer result/valence_roc_curve_fold_{fold}.png')
    plt.close()

    arousal_fpr, arousal_tpr, _ = roc_curve(arousal_true, predictions[:, 1])
    arousal_roc_auc = auc(arousal_fpr, arousal_tpr)
    arousal_auc_scores.append(arousal_roc_auc)
    plt.figure()
    plt.plot(arousal_fpr, arousal_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % arousal_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Arousal Fold {fold}')
    plt.legend(loc="lower right")
    plt.savefig(f'/Users/sudaxin/Desktop/EmoEst transformer result/arousal_roc_curve_fold_{fold}.png')
    plt.close()

    fold += 1

def average_classification_reports(reports):
    avg_report = {}
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):
            avg_report[key] = average_classification_reports([report[key] for report in reports])
        else:
            avg_report[key] = np.mean([report[key] for report in reports])
    return avg_report

average_mse = np.mean(mse_scores, axis=0)
average_rmse = np.mean(rmse_scores, axis=0)
average_mae = np.mean(mae_scores, axis=0)
average_r2 = np.mean(r2_scores, axis=0)
average_classification_accuracy = np.mean(classification_accuracy_scores, axis=0)

valence_report_avg = average_classification_reports(valence_reports)
arousal_report_avg = average_classification_reports(arousal_reports)

valence_report_avg_df = pd.DataFrame(valence_report_avg).transpose()
arousal_report_avg_df = pd.DataFrame(arousal_report_avg).transpose()

valence_report_avg_df.to_csv('/Users/sudaxin/Desktop/EmoEst transformer result/valence_classification_report_avg.csv')
arousal_report_avg_df.to_csv('/Users/sudaxin/Desktop/EmoEst transformer result/arousal_classification_report_avg.csv')

average_valence_auc = np.mean(valence_auc_scores)
average_arousal_auc = np.mean(arousal_auc_scores)
print(f'Average AUC for Valence: {average_valence_auc}')
print(f'Average AUC for Arousal: {average_arousal_auc}')

print("Cross-validation completed and results saved.")
print(f'Average MSE for Valence: {average_mse[0]}, Arousal: {average_mse[1]}')
print(f'Average RMSE for Valence: {average_rmse[0]}, Arousal: {average_rmse[1]}')
print(f'Average MAE for Valence: {average_mae[0]}, Arousal: {average_mae[1]}')
print(f'Average R2 for Valence: {average_r2[0]}, Arousal: {average_r2[1]}')
print(f'Average Classification Accuracy for Valence: {average_classification_accuracy[0]}, Arousal: {average_classification_accuracy[1]}')

metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2', 'Accuracy'],
    'Valence': [average_mse[0], average_rmse[0], average_mae[0], average_r2[0], average_classification_accuracy[0]],
    'Arousal': [average_mse[1], average_rmse[1], average_mae[1], average_r2[1], average_classification_accuracy[1]]
})

metrics_df.to_csv('/Users/sudaxin/Desktop/EmoEst transformer result/metrics_summary.csv', index=False)