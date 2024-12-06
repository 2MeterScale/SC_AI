import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# フォルダパス
train_file = 'output_vectors/vectors_CrabNet_train_data.csv'
data_folder = 'output_vectors'
output_folder = 'output_vectors/predicted_results'

# 出力フォルダの作成
os.makedirs(output_folder, exist_ok=True)

# データの読み込みと整形(学習用データは全て正例)
train_data = pd.read_csv(train_file)
train_data = train_data.drop_duplicates(subset=['composition_vector'])
train_data['composition_vector'] = train_data['composition_vector'].apply(eval)
X_train_list = train_data['composition_vector'].tolist()

# Zero-padding
X_train = pad_sequences(X_train_list, padding='post', dtype='float32')
maxlen = X_train.shape[1]

# マスクデータの作成関数
def mask_vector(vector, mask_ratio=0.3):
    masked = vector.copy()
    nonzero_indices = np.where(masked != 0)[0]
    # 非零要素のみをマスクするか、あるいは全長からランダムにマスク
    # ここでは簡単のために全長から選択
    indices = np.random.choice(len(masked), int(len(masked)*mask_ratio), replace=False)
    for idx in indices:
        masked[idx] = 0.0
    return masked

# モデル構築・学習
def build_and_train_mask_model(X_train):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Masking

    input_layer = Input(shape=(X_train.shape[1],))
    masked_input = Masking(mask_value=0.0)(input_layer)
    hidden_layer = Dense(64, activation='relu')(masked_input)
    output_layer = Dense(X_train.shape[1], activation=None)(hidden_layer)

    mask_model = Model(inputs=input_layer, outputs=output_layer)
    mask_model.compile(optimizer='adam', loss='mse')

    # 学習用マスクデータ作成
    X_train_masked = np.array([mask_vector(vec) for vec in X_train])
    mask_model.fit(X_train_masked, X_train, epochs=10, batch_size=16)
    return mask_model

# モデルの学習
mask_model = build_and_train_mask_model(X_train)

# 閾値計算
# 学習データで再構築誤差を計算し、その統計値から閾値を求める
X_train_masked = np.array([mask_vector(vec) for vec in X_train])
train_recon = mask_model.predict(X_train_masked)
train_errors = np.mean((X_train - train_recon)**2, axis=1)
threshold = np.mean(train_errors) + 2*np.std(train_errors)  # 平均+2σなど

# 超伝導体予測プログラム(自己教師あり学習ベース)
def predict_superconductivity(mask_model, data_folder, output_folder, threshold, maxlen):
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv') and filename != 'vectors_CrabNet_train_data.csv':
            file_path = os.path.join(data_folder, filename)
            data = pd.read_csv(file_path)

            if 'composition_vector' in data.columns:
                data['composition_vector'] = data['composition_vector'].apply(eval)
                X_test_list = data['composition_vector'].tolist()
                X_test = pad_sequences(X_test_list, padding='post', dtype='float32', maxlen=maxlen)

                # テストデータでも同様にマスクして再構築誤差を計算
                X_test_masked = np.array([mask_vector(vec) for vec in X_test])
                predictions = mask_model.predict(X_test_masked)
                test_errors = np.mean((X_test - predictions)**2, axis=1)

                # 閾値を用いて判定
                data['is_superconductor'] = (test_errors < threshold).astype(int)

                # 結果を保存
                output_file = os.path.join(output_folder, f"predicted_{filename}")
                data.to_csv(output_file, index=False)
                print(f"Predictions saved to: {output_file}")

# 予測を実施
predict_superconductivity(mask_model, data_folder, output_folder, threshold, maxlen)