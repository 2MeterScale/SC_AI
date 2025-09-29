import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter

API_KEY = "4e1LpV3Y9BIF4LLPaT8iL4coIVUnCIPF"  # 実際のAPIキーを入力

# CSV から組成データを読み込み
# ※ 例として1列目(0番目カラム)に組成が記載されていると仮定
df = pd.read_csv("unique_vectors_CrabNet_train_data.csv", header=None)
# df の 0番目カラムに組成文字列がある
compositions = df.iloc[:, 0].dropna().unique()  # 重複除去や欠損排除

with MPRester(API_KEY) as mpr:
    for formula in compositions:
        # 1) Materials Project で "pretty_formula" が一致するエントリを検索
        #    検索がヒットしない場合もあるため、結果が0件の場合はスキップする。
        results = mpr.query(
            {"pretty_formula": formula},
            ["material_id", "pretty_formula", "formula_anonymous", "spacegroup"]
        )

        if not results:
            print(f"✕ No entry found in MP for formula: {formula}")
            continue

        # 2) 複数候補があれば、ここでは便宜上最初の1件を採用
        entry = results[0]
        material_id = entry["material_id"]
        mp_formula = entry["pretty_formula"]
        sg = entry["spacegroup"]["symbol"] if "spacegroup" in entry else "UnknownSG"

        print(f"Found: {material_id} ({mp_formula}), SG={sg}")

        # 3) material_id を使って構造を取得
        structure = mpr.get_structure_by_material_id(material_id)

        # 4) CIFファイルとして保存
        cif_file_name = f"{material_id}_{mp_formula}.cif"
        writer = CifWriter(structure)
        writer.write_file(cif_file_name)

        print(f"  -> CIF saved to '{cif_file_name}'")