
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, ReLU, Layer
from tensorflow.keras.optimizers import Adam
from periodictable import elements

# -------------------
#  パラメータ設定
# -------------------
train_file = 'candidate_material_list_final/CrabNet_train_data.csv'
output_folder = 'output_vectors/generated_compositions'
os.makedirs(output_folder, exist_ok=True)

latent_dim = 64           # 潜在ベクトルの次元
max_atomic_number = 100   # 扱う原子番号の最大値（データに合わせて調整）
clip_value = 5.0          # 1つの元素カウントの上限 (適宜調整)

# 除外したい元素（例: N, Hg, Pb 等）
excluded_symbols = {
    # 毒性が高い元素
    "Be",  # ベリリウム（毒性）
    "As",  # ヒ素（毒性）
    "Cd",  # カドミウム（毒性）
    "Hg",  # 水銀（毒性）
    "Pb",  # 鉛（毒性）
    "Sb",  # アンチモン（毒性）
    "Tl",  # タリウム（毒性）
    "Se",  # セレン（過剰摂取で毒性）
    "Te",  # テルル（毒性）

    # 放射性元素（長寿命・短寿命を含む）
    "Po",  # ポロニウム
    "Rn",  # ラドン（気体）
    "Ra",  # ラジウム
    "Th",  # トリウム
    "U",   # ウラン
    "Np",  # ネプツニウム
    "Pu",  # プルトニウム
    "Am",  # アメリシウム
    "Cm",  # キュリウム
    "Bk",  # バークリウム
    "Cf",  # カリホルニウム
    "Es",  # アインスタイニウム
    "Fm",  # フェルミウム
    "Md",  # メンデレビウム
    "No",  # ノーベリウム
    "Lr",  # ローレンシウム
    "Rf",  # ラザホージウム
    "Db",  # ドブニウム
    "Sg",  # シーボーギウム
    "Bh",  # ボーリウム
    "Hs",  # ハッシウム
    "Mt",  # マイトネリウム
    "Ds",  # ダームスタチウム
    "Rg",  # レントゲニウム
    "Cn",  # コペルニシウム
    "Nh",  # ニホニウム
    "Fl",  # フレロビウム
    "Mc",  # モスコビウム
    "Lv",  # リバモリウム
    "Ts",  # テネシン
    "Og",  # オガネソン

    # ユーザ例にあった"N"（通常は毒性元素ではないが、要望に合わせて含む）
    "N"    # 窒素（単体としては安定だが、窒素酸化物など一部毒性化合物に注意）
}
excluded_atomic_numbers = [elements.symbol(sym).number for sym in excluded_symbols]

# -------------------
#  データ読み込み
# -------------------
train_data = pd.read_csv(train_file)
train_data = train_data.drop_duplicates(subset=['comp_atoms'])

# 文字列を辞書に変換
train_data['comp_atoms'] = train_data['comp_atoms'].apply(eval)

def atoms_to_vector(atoms_dict, max_atomic_number=max_atomic_number):
    """
    例: {57: 2, 29: 1, 8: 4} のような辞書をベクトルに変換。
    インデックス0が原子番号1、インデックス1が原子番号2、…となる。
    """
    vec = np.zeros(max_atomic_number, dtype=np.float32)
    for atomic_number, count in atoms_dict.items():
        if 1 <= atomic_number <= max_atomic_number:
            vec[atomic_number - 1] = count
    return vec

# -------------------
#  フィルタリング: 除外元素を含む行を落とす
# -------------------
def contains_excluded_atoms(atoms_dict, excluded_atomic_numbers):
    """
    'atoms_dict' 内にexcluded_atomic_numbers のいずれかが含まれていれば True
    """
    for atnum in atoms_dict.keys():
        if atnum in excluded_atomic_numbers:
            return True
    return False

# 毒性・窒素などを含む行を除去
filter_mask = train_data['comp_atoms'].apply(
    lambda d: not contains_excluded_atoms(d, excluded_atomic_numbers)
)
train_data_filtered = train_data[filter_mask].copy()

# ベクトル化
X_raw = np.array([
    atoms_to_vector(row) for row in train_data_filtered['comp_atoms']
])

# スケーリング（最大値スケーリング: 0～1）
max_val = X_raw.max() if X_raw.size > 0 else 1.0
if max_val == 0:
    max_val = 1.0
X_train = X_raw / max_val

# 学習データが全て除外されてしまった場合のガード
if X_train.shape[0] == 0:
    raise ValueError("全てのデータが除外され、学習できるサンプルがありません。excluded_symbols を見直してください。")

print(f"Filtered dataset size: {X_train.shape[0]}")

# -------------------
#  Generator 出力の除外元素を 0 にする層
# -------------------
class ExcludeElementsLayer(Layer):
    """
    生成器がどんな値を出しても、excluded_atomic_numbers に対応する出力を 0 に強制するカスタム層。
    """
    def __init__(self, excluded_atoms, **kwargs):
        super().__init__(**kwargs)
        self.excluded_atoms = excluded_atoms  # [7, 80, 82] 等の原子番号(1始まり)

    def call(self, inputs):
        # inputs: shape = (batch_size, max_atomic_number)
        # TensorFlow ではインデックス操作をするためにマスクを作る
        # atomic_number は 1始まり なので -1 して 0-based index にする
        mask = np.ones(inputs.shape[1], dtype=np.float32)
        for atnum in self.excluded_atoms:
            idx = atnum - 1  # 0-based index
            if 0 <= idx < inputs.shape[1]:
                mask[idx] = 0.0
        
        # maskをTFのTensorに変換
        mask_tensor = tf.constant(mask, dtype=inputs.dtype)
        # 各次元に対してマスクを掛け合わせ (excluded部分を0にする)
        return inputs * mask_tensor

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'excluded_atoms': self.excluded_atoms
        })
        return config


# -------------------
#  モデル構築 (無条件GAN)
# -------------------
from tensorflow.keras.layers import Input, Dense, LeakyReLU, ReLU

def build_generator(latent_dim, output_dim):
    """
    無条件GANのGenerator。
    出力はReLUで非負化し、clip_by_valueで上限を設定。
    さらに除外元素 (excluded_atomic_numbers) を強制0にするカスタム層を挟む。
    """
    inp = Input(shape=(latent_dim,))
    x = Dense(128)(inp)
    x = LeakyReLU(0.2)(x)
    x = Dense(output_dim)(x)
    x = ReLU()(x)
    # tf.clip_by_valueをLambda層でラップする
    x = tf.keras.layers.Lambda(lambda y: tf.clip_by_value(y, 0, clip_value))(x)  
    # 除外元素は 0 にセットする
    x = ExcludeElementsLayer(excluded_atomic_numbers)(x)
    model = Model(inp, x, name="Generator")
    return model

def build_discriminator(input_dim):
    """
    無条件GANのDiscriminator。
    """
    inp = Input(shape=(input_dim,))
    x = Dense(128)(inp)
    x = LeakyReLU(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out, name="Discriminator")
    return model

generator = build_generator(latent_dim, X_train.shape[1])
discriminator = build_discriminator(X_train.shape[1])

discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    # metrics=['accuracy']
)

# GANモデルを結合
z = Input(shape=(latent_dim,))
fake_data = generator(z)
# GAN学習時は識別器の重みを固定
discriminator.trainable = False
validity = discriminator(fake_data)
gan_model = Model(z, validity)
gan_model.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# -------------------
#  学習ループ
# -------------------
def train_gan(generator, discriminator, gan_model, X_train, 
              epochs=3000, batch_size=32, print_interval=500):
    half_batch = batch_size // 2
    valid_label = np.ones((half_batch, 1), dtype=np.float32)
    fake_label = np.zeros((half_batch, 1), dtype=np.float32)

    for epoch in range(epochs):
        # --- 1. Discriminatorの学習 ---
        # (a) 本物データ
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_data = X_train[idx]

        # (b) 偽物データ（Generator から生成）
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_data = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_data, valid_label)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- 2. Generatorの学習 ---
        # Generatorは「Discriminatorに本物と判断させたい」ので
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan_model.train_on_batch(noise, np.ones((batch_size, 1), dtype=np.float32))

        # 進捗表示
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} "
                  f"[D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] "
                  f"[G loss: {g_loss:.4f}]")

    # 学習後、モデルを保存
    generator.save("generator_for_superconductors_excl.h5")
    discriminator.save("discriminator_for_superconductors_excl.h5")
    print("Models saved.")

# -------------------
#  学習開始
# -------------------
train_gan(generator, discriminator, gan_model, X_train,
          epochs=3000, batch_size=32, print_interval=500)

# -------------------
#  組成式生成
# -------------------
def vector_to_formula(vector):
    """
    ベクトル -> 組成式文字列の変換。
    小数は四捨五入して整数へ（0 なら元素表記しない）。
    """
    formula_parts = []
    for atomic_number, val in enumerate(vector, start=1):
        if atomic_number > len(elements):
            break
        count = int(round(val))
        if count > 0:
            elem_symbol = elements[atomic_number].symbol
            if count == 1:
                formula_parts.append(elem_symbol)
            else:
                formula_parts.append(f"{elem_symbol}{count}")
    return "".join(formula_parts)

def generate_new_compositions(generator, n_samples=10):
    """
    Generator から n_samples 個の新規組成を生成し、構造式に変換して返す
    """
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    gen_data_scaled = generator.predict(noise)
    # 学習時のスケーリング (/ max_val) を戻す
    gen_data = gen_data_scaled * max_val

    compositions = []
    for vec in gen_data:
        formula = vector_to_formula(vec)
        compositions.append(formula)
    return compositions

# -------------------
#  サンプル生成
# -------------------
new_formulas = generate_new_compositions(generator, n_samples=10)
print("=== Generated Compositions (excluded N, Hg, Pb) ===")
for i, f in enumerate(new_formulas, start=1):
    print(f"{i}: {f}")