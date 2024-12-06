import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
import tensorflow as tf

def build_contrastive_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32)(x)
    # tf.math.l2_normalize を Lambda レイヤー内で使用
    outputs = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    
    model = Model(inputs, outputs)
    return model
# フォルダパス
train_file = 'output_vectors/vectors_CrabNet_train_data.csv'
data_folder = 'output_vectors'
output_folder = 'output_vectors/predicted_results'
os.makedirs(output_folder, exist_ok=True)

# データ読み込み・整形
train_data = pd.read_csv(train_file)
train_data = train_data.drop_duplicates(subset=['composition_vector'])
train_data['composition_vector'] = train_data['composition_vector'].apply(eval)
X_train_list = train_data['composition_vector'].tolist()

# ゼロパディング
X_train = pad_sequences(X_train_list, padding='post', dtype='float32')
maxlen = X_train.shape[1]

# マスク(データ拡張)関数
def mask_vector(vector, mask_ratio=0.3):
    masked = vector.copy()
    # 全要素からランダムにマスク
    indices = np.random.choice(len(masked), int(len(masked)*mask_ratio), replace=False)
    for idx in indices:
        masked[idx] = 0.0
    return masked

# コントラスト学習モデル構築
# シンプルな埋め込みモデル: 入力 -> Dense(64, relu) -> Dense(32)出力を埋め込みとする
def build_contrastive_model(input_dim):
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Dense(64, activation='relu')(x)
    # L2正規化された埋め込みベクトルを出力 (SimCLR風)
    x = layers.Dense(32)(x)
    outputs = tf.nn.l2_normalize(x, axis=1)
    model = Model(inputs, outputs)
    return model

contrast_model = build_contrastive_model(maxlen)

# NT-Xent loss (SimCLRで用いられるコントラスト損失)
# 下記はシンプルな実装例
def nt_xent_loss(z1, z2, temperature=0.07):
    # z1, z2: (batch_size, embedding_dim)
    batch_size = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0) # (2*batch_size, emb_dim)

    # 類似度行列: z z^T
    similarity_matrix = tf.matmul(z, z, transpose_b=True)
    # 同一ベクトル同士の類似度を除去するためにマスクを作成
    mask = tf.eye(2*batch_size)
    # 類似度を温度でスケーリング
    similarity_matrix = similarity_matrix / temperature

    # 正例ペアは (i, i+batch_size) と (i+batch_size, i)
    # 対角ブロックを除きつつ、i番目サンプルに対する正例はi+batch_size位置
    # ラベルはiの正例はi+batch_size(もしi<batch_sizeなら)
    labels = tf.range(batch_size)
    labels = tf.concat([labels+batch_size, labels], axis=0) # 正例インデックス

    # log_softmaxを計算
    # 分母は全てのペア(自分自身を除く)、分子は対応する正例ペア
    exp_sim = tf.exp(similarity_matrix)
    exp_sim_sum = tf.reduce_sum(exp_sim * (1 - mask), axis=1)
    pos_sim = tf.exp(tf.gather_nd(similarity_matrix, tf.stack([tf.range(2*batch_size), labels], axis=1)))

    loss = -tf.reduce_mean(tf.math.log(pos_sim / exp_sim_sum))
    return loss

# データをバッチ化
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(len(X_train)).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(1e-3)

# コントラスト学習ループ (簡易的な手動学習ループ)
epochs = 10
for epoch in range(epochs):
    losses = []
    for batch in dataset:
        # 同一バッチ内で2種類のマスク拡張を行い、ペアを作る
        x1 = np.array([mask_vector(v) for v in batch.numpy()])
        x2 = np.array([mask_vector(v) for v in batch.numpy()])

        with tf.GradientTape() as tape:
            z1 = contrast_model(x1, training=True)
            z2 = contrast_model(x2, training=True)
            loss_value = nt_xent_loss(z1, z2)
        grads = tape.gradient(loss_value, contrast_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, contrast_model.trainable_variables))
        losses.append(loss_value.numpy())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")

# 学習後、トレーニングデータを埋め込みに変換して分布を確認
Z_train = contrast_model(X_train, training=False).numpy()
mean_emb = np.mean(Z_train, axis=0)
dists = np.linalg.norm(Z_train - mean_emb, axis=1)
threshold = np.mean(dists) + 2*np.std(dists)  # 例えば平均+2σを閾値とする

# 予測関数：テストデータを埋め込みに変換し、mean_embからの距離で判定
def predict_superconductivity(contrast_model, data_folder, output_folder, mean_emb, threshold, maxlen):
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv') and filename != 'vectors_CrabNet_train_data.csv':
            file_path = os.path.join(data_folder, filename)
            data = pd.read_csv(file_path)

            if 'composition_vector' in data.columns:
                data['composition_vector'] = data['composition_vector'].apply(eval)
                X_test_list = data['composition_vector'].tolist()
                X_test = pad_sequences(X_test_list, padding='post', dtype='float32', maxlen=maxlen)

                Z_test = contrast_model(X_test, training=False).numpy()
                test_dists = np.linalg.norm(Z_test - mean_emb, axis=1)
                data['is_superconductor'] = (test_dists < threshold).astype(int)

                output_file = os.path.join(output_folder, f"predicted_{filename}")
                data.to_csv(output_file, index=False)
                print(f"Predictions saved to: {output_file}")

# 推論実行
predict_superconductivity(contrast_model, data_folder, output_folder, mean_emb, threshold, maxlen)