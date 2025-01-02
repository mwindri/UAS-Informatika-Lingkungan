# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

# Baca Data dari CSV
df = pd.read_csv('gempa_bumi.csv')

# Pilih Kolom yang Sesuai dengan Data (tanpa d+1)
data = df[['+OT(min)', 'Kdlmn(Km)', 'PhaseCount', 'Mag', 'MagCount']]  # Kolom yang relevan
data = data.dropna()  # Menghapus baris kosong jika ada

# Target diubah menjadi kolom MagCount untuk prediksi
target_column = 'MagCount'

# Train-Test Split
train = data[:int(0.8 * len(data))]
test = data[int(0.8 * len(data)):]

# Pilih Fitur (X) dan Target (y)
x_train = train[['+OT(min)', 'Kdlmn(Km)', 'PhaseCount', 'Mag']]
y_train = train[target_column]
x_test = test[['+OT(min)', 'Kdlmn(Km)', 'PhaseCount', 'Mag']]
y_test = test[target_column]

# Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# KNN dengan GridSearchCV
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# Fit dan Prediksi
model.fit(x_train_scaled, y_train)
preds = model.predict(x_test_scaled)

# Hitung RMSE
rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
print(f"RMSE: {rms}")

# Simpan Hasil Prediksi ke DataFrame
test = test.copy()  # Pastikan test adalah salinan independen
test['Predictions'] = preds

# Simpan Hasil ke CSV
test[['+OT(min)', 'Kdlmn(Km)', 'PhaseCount', 'Mag', 'MagCount', 'Predictions']].to_csv('predicted_results.csv', index=False)
print("Hasil prediksi disimpan ke 'predicted_results.csv'")

# Plot Hasil
plt.figure(figsize=(14, 7))
plt.plot(test[target_column].values, label=f'Actual {target_column}', color='blue')
plt.plot(test['Predictions'].values, label=f'Predicted {target_column}', color='red')
plt.title(f"Actual vs Predicted {target_column}")
plt.xlabel("Data Gempa Bumi 2024")
plt.ylabel(f"{target_column} Value")
plt.legend()
plt.show()
