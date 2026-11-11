import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Load data
try:
    df = pd.read_csv('Updated Quality of Life Data.csv', sep=';')
except FileNotFoundError:
    print("File 'Updated Quality of Life Data.csv' tidak ditemukan.")
    exit()

# Preprocessing: Buat target klasifikasi 'Panjang_Umur'
df['Panjang_Umur'] = np.where(df['age_at_death'] >= 80, 'Ya', 'Tidak')

# Ambil 11 data pertama (ID 10001 - 10011)
df_11 = df.iloc[0:11].copy()

# Drop kolom yang tidak relevan (id, age_at_death)
if 'id' in df_11.columns:
    df_11 = df_11.drop(columns=['id', 'age_at_death'])

# Ubah teks kategori (gender, occupation_type) ke angka agar bisa masuk model
df_11 = pd.get_dummies(df_11, columns=['gender', 'occupation_type'])

# Split data: 10 baris pertama untuk train, baris ke-11 untuk test
train_data = df_11.iloc[:10]
X_train = train_data.drop(columns=['Panjang_Umur'])
y_train = train_data['Panjang_Umur']

test_data = df_11.iloc[10:11]
X_test = test_data.drop(columns=['Panjang_Umur'])
y_test = test_data['Panjang_Umur'].values[0]

# Setup & train model (5 pohon, max fitur 1, max depth 1)
rf = RandomForestClassifier(n_estimators=5, max_features=1, max_depth=1, random_state=42)
rf.fit(X_train, y_train)

# Prediksi
pred = rf.predict(X_test)

# Print hasil
print("\n--- Hasil Prediksi Data Uji ---")
for col in X_test.columns:
    print(f"{col}: {X_test.iloc[0][col]}")

print(f"\nTarget Aktual : {y_test}")
print(f"Prediksi      : {pred[0]}")

# Visualisasi pohon
plt.figure(figsize=(20, 5))
for i, tree in enumerate(rf.estimators_):
    plt.subplot(1, 5, i+1)
    plot_tree(tree, 
              feature_names=list(X_train.columns), 
              class_names=list(map(str, rf.classes_)), 
              filled=True, 
              rounded=True, 
              fontsize=9)
    plt.title(f'Tree {i+1}')

plt.tight_layout()
plt.show()