import pandas as pd
import ast

# CSVファイルのパス
file_path = "candidate_material_list_final/CrabNet_train_data.csv"
output_file_path = "candidate_material_list_final/CrabNet_train_data_AC.csv"  # 出力ファイルのパス

# CSVファイルを読み込む
df = pd.read_csv(file_path)

# AC列のデータを2×nの配列として変換
def parse_ac_column(value):
    try:
        # 文字列を辞書に変換
        data_dict = ast.literal_eval(value)
        # 辞書を2×nのリストに変換
        return [[k, v] for k, v in data_dict.items()]
    except:
        return None  # 変換できない場合はNoneを返す

# 変換を適用
df["AC_parsed"] = df[28].apply(parse_ac_column)

# 新しいCSVファイルに保存
df.to_csv(output_file_path, index=False)

# 出力ファイルのパスを表示
print(f"変換後のデータをCSVファイルに保存しました: {output_file_path}")