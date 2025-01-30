import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -----------------------------
# デバイス設定
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available()
                      else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# =========================================================
# 1. CSV から 3,4 元素系に仕分ける関数
# =========================================================
def load_composition_vectors(csv_path):
    """
    CSVファイルを読み込み、
    - 長さ6 (3元素系)
    - 長さ8 (4元素系)
    のデータをそれぞれリストに分けて返す。
    それ以外の長さのベクトルは無視。
    """
    data_6dim = []
    data_8dim = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec_str = row['composition_vector'].strip()
            vec_str = vec_str.replace('[', '').replace(']', '')
            vec_list = vec_str.split(',')
            # float変換
            vec_floats = [float(x.strip()) for x in vec_list]

            length = len(vec_floats)
            if length == 6:   # 3元素系
                data_6dim.append(vec_floats)
            elif length == 8: # 4元素系
                data_8dim.append(vec_floats)
            else:
                # 6,8以外は無視
                pass

    return data_6dim, data_8dim


# =========================================================
# 2. PyTorch Dataset
# =========================================================
class CompositionDataset(Dataset):
    def __init__(self, data_list):
        self.data = np.array(data_list, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# =========================================================
# 3. Discriminator, Generator
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim=16, output_dim=8, hidden_dim=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.model(z)


# =========================================================
# 4. 簡易GAN学習関数
# =========================================================
def train_gan(dataloader, generator, discriminator,
              g_optimizer, d_optimizer, criterion,
              z_dim=16, epochs=50):
    for epoch in range(epochs):
        for real_data in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # --- 識別器(D)の学習 ---
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 本物データ
            d_out_real = discriminator(real_data)
            d_loss_real = criterion(d_out_real, real_labels)

            # 偽データ
            z = torch.randn(batch_size, z_dim, device=device)
            fake_data = generator(z)
            d_out_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(d_out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --- 生成器(G)の学習 ---
            z = torch.randn(batch_size, z_dim, device=device)
            fake_data = generator(z)
            d_out_fake_for_g = discriminator(fake_data)
            # Generatorにとっては 1 (本物) と判定されたい
            g_loss = criterion(d_out_fake_for_g, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    print("学習完了！")


# =========================================================
# 5. 後処理 (正の整数化、使いたくない元素の除外、酸素数制限 など)
# =========================================================
def postprocess_composition(vec,
                            excluded_z,
                            min_count=1, max_count=20,
                            oxygen_min=5, oxygen_max=8):
    """
    出力ベクトル(連続値)を以下の制約で後処理:
      1) 負値→0→丸め→最低1 で正の整数化
      2) 原子番号がexcluded_zに含まれる場合→最寄りのallowed_zに丸め
      3) 酸素(O=8)の個数を[oxygen_min, oxygen_max]にクリップ
      4) 個数スロットを[min_count, max_count]にクリップ

    偶数インデックス(0-based)が原子番号(Z), 奇数インデックスが個数(c) の想定。
    """
    out = vec.copy()

    # (A) 負値を0に, 丸めて整数化
    out = np.maximum(out, 0.0)
    out = np.round(out)

    # (B) 最小1
    out = np.where(out < 1, 1, out)

    # (C) 許容する原子番号(1~118)からexcluded_z除外
    all_z = set(range(1, 119))
    allowed_z = sorted(list(all_z - excluded_z))

    # (D) 原子番号スロットを修正
    for idx in range(0, len(out), 2):
        z = int(out[idx])
        if z not in allowed_z:
            # 近いものに置き換え
            nearest_z = min(allowed_z, key=lambda x: abs(x - z))
            out[idx] = nearest_z

    # (E) 個数スロットを[min_count, max_count]でクリップ
    for idx in range(1, len(out), 2):
        out[idx] = np.clip(out[idx], min_count, max_count)

    # (F) 酸素(O=8)の個数を[oxygen_min, oxygen_max]にクリップ
    for idx in range(0, len(out), 2):
        if int(out[idx]) == 8:  # O
            out[idx+1] = np.clip(out[idx+1], oxygen_min, oxygen_max)

    return out.astype(int)


# =========================================================
# 6. 原子番号 → 元素記号 変換用の辞書 (一例)
#    辞書に無い番号は "E{番号}" として扱う簡易実装。
# =========================================================
element_symbols = {
    1:   "H",   2:   "He",  3:   "Li",  4:   "Be",  5:   "B",   6:   "C",
    7:   "N",   8:   "O",   9:   "F",   10:  "Ne",  11:  "Na",  12:  "Mg",
    13:  "Al",  14:  "Si",  15:  "P",   16:  "S",   17:  "Cl",  18:  "Ar",
    19:  "K",   20:  "Ca",  21:  "Sc",  22:  "Ti",  23:  "V",   24:  "Cr",
    25:  "Mn",  26:  "Fe",  27:  "Co",  28:  "Ni",  29:  "Cu",  30:  "Zn",
    31:  "Ga",  32:  "Ge",  33:  "As",  34:  "Se",  35:  "Br",  36:  "Kr",
    37:  "Rb",  38:  "Sr",  39:  "Y",   40:  "Zr",  41:  "Nb",  42:  "Mo",
    43:  "Tc",  44:  "Ru",  45:  "Rh",  46:  "Pd",  47:  "Ag",  48:  "Cd",
    49:  "In",  50:  "Sn",  51:  "Sb",  52:  "Te",  53:  "I",   54:  "Xe",
    55:  "Cs",  56:  "Ba",  57:  "La",  58:  "Ce",  59:  "Pr",  60:  "Nd",
    61:  "Pm",  62:  "Sm",  63:  "Eu",  64:  "Gd",  65:  "Tb",  66:  "Dy",
    67:  "Ho",  68:  "Er",  69:  "Tm",  70:  "Yb",  71:  "Lu",  72:  "Hf",
    73:  "Ta",  74:  "W",   75:  "Re",  76:  "Os",  77:  "Ir",  78:  "Pt",
    79:  "Au",  80:  "Hg",  81:  "Tl",  82:  "Pb",  83:  "Bi",  84:  "Po",
    85:  "At",  86:  "Rn",  87:  "Fr",  88:  "Ra",  89:  "Ac",  90:  "Th",
    91:  "Pa",  92:  "U",   93:  "Np",  94:  "Pu",  95:  "Am",  96:  "Cm",
    97:  "Bk",  98:  "Cf",  99:  "Es",  100: "Fm",  101: "Md",  102: "No",
    103: "Lr",  104: "Rf",  105: "Db",  106: "Sg",  107: "Bh",  108: "Hs",
    109: "Mt",  110: "Ds",  111: "Rg",  112: "Cn",  113: "Nh",  114: "Fl",
    115: "Mc",  116: "Lv",  117: "Ts",  118: "Og"
}


def convert_vector_to_formula(vec_int):
    """
    例: vec_int = [Z1, c1, Z2, c2, ...] を
         "Element1{c1}Element2{c2}..." のような化学式文字列に変換
         ただし c=1 のときは省略 (例: "Cu" など)
         辞書に無い原子番号は "E{Z}" とする
    """
    formula_parts = []
    for i in range(0, len(vec_int), 2):
        z = int(vec_int[i])
        c = int(vec_int[i+1])
        # 元素記号を辞書から引く (無い場合は E{z})
        if z in element_symbols:
            symbol = element_symbols[z]
        else:
            symbol = f"E{z}"  # fallback
        # 個数が1なら省略、1以上なら数字をつける
        if c == 1:
            part = f"{symbol}"
        else:
            part = f"{symbol}{c}"
        formula_parts.append(part)
    return "".join(formula_parts)


# =========================================================
# 7. メイン処理
# =========================================================
def main():
    # ----------------------------------------------------
    # (A) CSV から 3,4元素系 データを仕分けて取得
    # ----------------------------------------------------
    csv_path = "unique_vectors_CrabNet_train_data.csv"
    data_6dim, data_8dim = load_composition_vectors(csv_path)

    print(f"3元素系(6次元)データ数: {len(data_6dim)}")
    print(f"4元素系(8次元)データ数: {len(data_8dim)}")

    # ----------------------------------------------------
    # (B) 使用したくない元素(原子番号)の定義
    # ----------------------------------------------------
    # 1) 希ガス: He(2), Ne(10), Ar(18), Kr(36), Xe(54), Rn(86), Og(118)
    # 2) 窒素: N(7)
    # 3) 水銀: Hg(80)
    # 4) 放射性(簡易): Tc(43), Pm(61), & 84以上
    # 5) アルカリ土類: Be(4), Mg(12), Ca(20), Sr(38), Ba(56), Ra(88)
    excluded_z = {2, 7, 10, 18, 36, 54, 80, 43, 61, 3, 11, 19, 37, 55} | set(range(84, 119))

    # 後処理パラメータ
    oxygen_min, oxygen_max = 5, 8
    min_count, max_count = 1, 20

    # ----------------------------------------------------
    # (C) 3元素系(6次元) のGAN学習 & 生成
    # ----------------------------------------------------
    if len(data_6dim) > 0:
        dataset_3elem = CompositionDataset(data_6dim)
        dataloader_3elem = DataLoader(dataset_3elem, batch_size=4, shuffle=True)

        z_dim_3 = 16
        G_3 = Generator(z_dim=z_dim_3, output_dim=6, hidden_dim=32).to(device)
        D_3 = Discriminator(input_dim=6, hidden_dim=32).to(device)

        lr = 1e-3
        g_optimizer_3 = optim.Adam(G_3.parameters(), lr=lr)
        d_optimizer_3 = optim.Adam(D_3.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print("\n===== 3元素系(6次元) GAN 学習開始 =====")
        train_gan(dataloader_3elem, G_3, D_3,
                  g_optimizer_3, d_optimizer_3,
                  criterion, z_dim=z_dim_3, epochs=50)

        # -------- 200個生成してCSVに書き出す --------
        output_file_3 = "generated_3elem.csv"
        G_3.eval()
        num_samples = 200

        z = torch.randn(num_samples, z_dim_3).to(device)
        with torch.no_grad():
            gen_3_raw = G_3(z).cpu().numpy()

        with open(output_file_3, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["formula_3elements"])  # ヘッダ
            for vec in gen_3_raw:
                # 後処理(整数化, 除外元素排除, 酸素数制限 etc)
                processed = postprocess_composition(
                    vec,
                    excluded_z=excluded_z,
                    oxygen_min=oxygen_min,
                    oxygen_max=oxygen_max,
                    min_count=min_count,
                    max_count=max_count
                )
                # 組成式を文字列化
                formula = convert_vector_to_formula(processed)
                writer.writerow([formula])

        print(f"\n→ 3元素系: 合計 {num_samples}件の組成式を {output_file_3} に保存しました。")

    else:
        print("\n3元素系(6次元)のデータはありません。")


    # ----------------------------------------------------
    # (D) 4元素系(8次元) のGAN学習 & 生成
    # ----------------------------------------------------
    if len(data_8dim) > 0:
        dataset_4elem = CompositionDataset(data_8dim)
        dataloader_4elem = DataLoader(dataset_4elem, batch_size=4, shuffle=True)

        z_dim_4 = 16
        G_4 = Generator(z_dim=z_dim_4, output_dim=8, hidden_dim=32).to(device)
        D_4 = Discriminator(input_dim=8, hidden_dim=32).to(device)

        lr = 1e-3
        g_optimizer_4 = optim.Adam(G_4.parameters(), lr=lr)
        d_optimizer_4 = optim.Adam(D_4.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print("\n===== 4元素系(8次元) GAN 学習開始 =====")
        train_gan(dataloader_4elem, G_4, D_4,
                  g_optimizer_4, d_optimizer_4,
                  criterion, z_dim=z_dim_4, epochs=50)

        # -------- 200個生成してCSVに書き出す --------
        output_file_4 = "generated_4elem.csv"
        G_4.eval()
        num_samples = 200

        z = torch.randn(num_samples, z_dim_4).to(device)
        with torch.no_grad():
            gen_4_raw = G_4(z).cpu().numpy()

        with open(output_file_4, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["formula_4elements"])  # ヘッダ
            for vec in gen_4_raw:
                # 後処理(整数化, 除外元素排除, 酸素数 etc)
                processed = postprocess_composition(
                    vec,
                    excluded_z=excluded_z,
                    oxygen_min=oxygen_min,
                    oxygen_max=oxygen_max,
                    min_count=min_count,
                    max_count=max_count
                )
                # 組成式を文字列化
                formula = convert_vector_to_formula(processed)
                writer.writerow([formula])

        print(f"\n→ 4元素系: 合計 {num_samples}件の組成式を {output_file_4} に保存しました。")

    else:
        print("\n4元素系(8次元)のデータはありません。")


if __name__ == "__main__":
    main()