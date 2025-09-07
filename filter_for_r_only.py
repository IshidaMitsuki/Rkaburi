# =============================================================================
# SCRIPT NAME: filter_for_r_only.py
#
# PURPOSE:
#   完全なデータセットから、R言語に関連する衝突データのみを抽出し、
#   軽量な分析用CSVファイルを生成する。
# =============================================================================
import pandas as pd
import os

# --- CONFIG ---
# 読み込む元の、完全なデータセットファイル
INPUT_FILENAME = "R/final_dataset_for_analysis.csv"

# 出力する、R言語に絞った軽量なデータセットファイル
OUTPUT_DIRECTORY = "R"
OUTPUT_FILENAME_R_ONLY = "r_focused_dataset.csv"
# ----------------

if __name__ == "__main__":
    print(f"'{INPUT_FILENAME}' を読み込んでいます...")
    try:
        df_raw = pd.read_csv(INPUT_FILENAME)
        print(f"✅ データ読み込み完了。全 {len(df_raw)} 行。")
    except FileNotFoundError:
        print(f"❌ エラー: '{INPUT_FILENAME}' が見つかりません。")
        exit()

    # --- フィルタリング処理 ---
    print("\nR言語関連のデータを抽出中...")
    
    # 抽出条件:
    # 1. 衝突相手の言語 (language) が 'R' である
    # 2. または、衝突がなかった (conflict_type が 'none')
    condition = (df_raw['language'] == 'R') | (df_raw['conflict_type'] == 'none')
    
    df_r_focused = df_raw[condition].copy()
    
    print(f"✅ 抽出完了。{len(df_r_focused)} 行のR関連データを検出しました。")
    
    # --- ファイル保存 ---
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME_R_ONLY)
    df_r_focused.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ R言語に絞った軽量版データセットを '{output_path}' に保存しました。")