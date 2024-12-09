import os
import pandas as pd
import numpy as np
from xenonpy.descriptor import CompositionFeaturizer
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras import layers, models

data_folder = 'output_vectors'
train_file = 'vectors_CrabNet_train_data.csv'
output_folder = 'output_vectors/picked_closest'
os.makedirs(output_folder, exist_ok=True)

# 1. 全CSVファイルからcomposition列を読み込み、XenonPyでベクトル化
featurizer = CompositionFeaturizer()

all_data_vectors = {}  # filename -> dict with keys: 'compositions', 'vectors'
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        if 'composition' in df.columns:
            comps = df['composition'].dropna().unique()
            # ベクトル化
            X = featurizer.transform(comps)
            X = np.nan_to_num(X)  # NaNを0埋め
            all_data_vectors[filename] = {
                'compositions': comps,
                'vectors': X
            }

# 2. train_fileから取得したベクトルでGANを学習
train_data = all_data_vectors[train_file]['vectors']
feature_dim = train_data.shape[1]

# 簡易的なGAN構築
def build_generator(z_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(z_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim)  # 出力は線形
    ])
    return model

def build_discriminator(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

z_dim = 64
generator = build_generator(z_dim, feature_dim)
discriminator = build_discriminator(feature_dim)

cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_opt = tf.keras.optimizers.Adam(1e-3)
disc_opt = tf.keras.optimizers.Adam(1e-3)

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(train_data.astype('float32')).shuffle(len(train_data)).batch(batch_size)

epochs = 5  # 必要に応じて増やす

@tf.function
def train_step(real_data):
    bsz = tf.shape(real_data)[0]
    noise = tf.random.normal(shape=(bsz, z_dim))

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        fake_data = generator(noise, training=True)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2.0

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    disc_opt.apply_gradients(zip(grads_disc, discriminator.trainable_variables))
    gen_opt.apply_gradients(zip(grads_gen, generator.trainable_variables))

    return gen_loss, disc_loss

for epoch in range(epochs):
    g_losses = []
    d_losses = []
    for real_batch in dataset:
        gl, dl = train_step(real_batch)
        g_losses.append(gl.numpy())
        d_losses.append(dl.numpy())
    print(f"Epoch {epoch+1}/{epochs} Gen Loss: {np.mean(g_losses):.4f}, Disc Loss: {np.mean(d_losses):.4f}")

# 3. GANで類似ベクトルを生成
num_generated = 10
noise = tf.random.normal(shape=(num_generated, z_dim))
generated_vectors = generator(noise, training=False).numpy()

# 4. "vectors_CrabNet_train_data.csv"以外のファイルから、生成ベクトルに近いデータをピックアップ
other_files = [f for f in all_data_vectors.keys() if f != train_file]

picked_results = []
for vec in generated_vectors:
    # 全ての他ファイルのベクトルを結合して検索対象とする
    # ファイル名情報を持ちたいので、一旦ファイルごとにNearestNeighborsで探索し、最も近いものを取得
    closest_dist = float('inf')
    closest_comp = None
    closest_file = None
    for fname in other_files:
        X_other = all_data_vectors[fname]['vectors']
        comps_other = all_data_vectors[fname]['compositions']

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_other)
        dist, idx = nn.kneighbors(vec.reshape(1, -1))
        if dist[0][0] < closest_dist:
            closest_dist = dist[0][0]
            closest_comp = comps_other[idx[0][0]]
            closest_file = fname

    picked_results.append((vec.tolist(), closest_file, closest_comp, closest_dist))

# 結果を保存
results_df = pd.DataFrame(picked_results, columns=['generated_vector', 'closest_file', 'closest_composition', 'distance'])
output_file = os.path.join(output_folder, 'picked_closest_results.csv')
results_df.to_csv(output_file, index=False)
print(f"Picked closest results saved to {output_file}")