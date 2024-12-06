import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

# データフォルダとファイルパス
train_file = 'output_vectors/vectors_CrabNet_train_data.csv'
data_folder = 'output_vectors'

# 学習用データの読み込み
train_data = pd.read_csv(train_file)

# 重複を削除
train_data = train_data.drop_duplicates(subset=['composition_vector'])

# ベクトルデータを整形
train_data['composition_vector'] = train_data['composition_vector'].apply(eval)  # 文字列をリストに変換
mlb = MultiLabelBinarizer()  # ベクトルをバイナリ形式に変換
X_train = mlb.fit_transform(train_data['composition_vector'])

# 全て正例（超伝導）としてターゲットを設定
y_train = [1] * len(train_data)

# ロジスティック回帰モデルの学習
model = LogisticRegression()
model.fit(X_train, y_train)

# フォルダ内の他のファイルを処理
for filename in os.listdir(data_folder):
    if filename.endswith('.csv') and filename != 'vectors_CrabNet_train_data.csv':
        file_path = os.path.join(data_folder, filename)
        data = pd.read_csv(file_path)

        # ベクトルデータを整形
        if 'composition_vector' in data.columns:
            data['composition_vector'] = data['composition_vector'].apply(eval)  # 文字列をリストに変換
            X_test = mlb.transform(data['composition_vector'])

            # 判定の実行
            predictions = model.predict(X_test)
            data['is_superconductor'] = predictions

            # 結果を保存
            output_file = os.path.join(data_folder, f"predicted_{filename}")
            data.to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")