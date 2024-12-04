import csv

# 入力ファイルと出力ファイルのパス
input_file = '20240322_MDR_OAndM.txt'
output_file = 'oxide_metallic_data_with_spacegroup.csv'

# Oxide & Metallic の重要なカラム（PDFの情報を参照）
columns = [
    "num",  # data number
    "refno",  # reference number
    "commt",  # comment
    "name",  # common formula of materials
    "element",  # chemical formula
    "ma1", "ma2",  # element name of materials and composition (MA1)
    "mb1", "mb2",  # element name of materials and composition (MA2)
    "mc1", "mc2",  # element name of materials and composition (MA3)
    "mo1", "oz",  # oxygen and measured value of oxygen content
    "shape",  # shape
    "year",  # publication year
    "t1", "t2", "t3",  # transition temperatures (R=0, midpoint, R=100%)
    "moment",  # magnetic moment per formula
    "dens",  # density
    "lata", "latb", "latc",  # lattice constants
    "str1", "str3",  # crystal structure and common name of structure
    "spaceg"  # space group
]

# 入力ファイルを読み込み、CSVに整形
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')  # タブ区切りでデータを読み込む
    writer = csv.writer(outfile)
    
    # ヘッダーを書き込む
    writer.writerow(columns)
    
    # データ行を処理
    for row in reader:
        # 必要なカラムを抽出
        try:
            filtered_row = [
                row[0],  # num
                row[1],  # refno
                row[2],  # commt
                row[3],  # name
                row[4],  # element
                row[6], row[7],  # ma1, ma2
                row[8], row[9],  # mb1, mb2
                row[10], row[11],  # mc1, mc2
                row[27], row[28],  # mo1, oz
                row[29],  # shape
                row[30],  # year
                row[85], row[86], row[87],  # t1, t2, t3
                row[37],  # moment
                row[41],  # dens
                row[71], row[72], row[73],  # lata, latb, latc
                row[66], row[67],  # str1, str3
                row[68]  # spaceg
            ]
            writer.writerow(filtered_row)
        except IndexError:
            # データが不足している場合はスキップ
            continue

print(f"整形されたデータが'{output_file}'に保存されました。")