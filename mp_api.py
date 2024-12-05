from mp_api.client import MPRester

# 取得したAPIキーを入力してください
API_KEY = "4e1LpV3Y9BIF4LLPaT8iL4coIVUnCIPF"

# MPResterクライアントを初期化
with MPRester(API_KEY) as mpr:
    # 無機物質のデータを取得
    # 例: SiとOを含む物質の情報を取得
    results = mpr.materials.search(elements=["Si", "O"], num_elements=2)

    # 取得したデータを表示
    for material in results:
        print(f"Material ID: {material.material_id}")
        print(f"Formula: {material.formula_pretty}")
        print(f"Band Gap: {material.band_gap}")
        print(f"Density: {material.density}")
        print(f"Crystal System: {material.crystal_system}")
        print("-" * 40)