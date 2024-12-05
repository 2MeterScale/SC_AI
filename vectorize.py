import os
import pandas as pd
import re
from periodictable import elements

# フォルダパスを指定
folder_path = '/candidate_material_list_final'
output_folder = '/output_vectors'

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 元素記号から原子番号への対応辞書を作成
element_to_atomic_number = {str(e.symbol): e.number for e in elements if e.number is not None}

# 組成式をベクトルに変換する関数
def formula_to_vector(formula):
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.*\d*)', formula)
    vector = []
    for element, count in matches:
        atomic_number = element_to_atomic_number.get(element)
        if atomic_number is not None:
            count = float(count) if count else 1.0  # 数が省略されている場合は1
            vector.append((atomic_number, count))
    vector = [item for pair in sorted(vector) for item in pair]  # ソートしてフラットなリストに変換
    return vector

# フォルダ内の全てのCSVファイルを処理
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        output_file_path = os.path.join(output_folder, f'vectors_{filename}')

        # CSVファイルを読み込む
        data = pd.read_csv(file_path)

        # 必要な列があるか確認
        if 'new_formula' in data.columns:
            # ベクトル化を実行
            data['composition_vector'] = data['new_formula'].apply(formula_to_vector)
            
            # 結果を保存
            data[['new_formula', 'composition_vector']].to_csv(output_file_path, index=False)
            print(f"Processed and saved: {output_file_path}")
        else:
            print(f"Skipped: {filename} (missing 'new_formula' column)")