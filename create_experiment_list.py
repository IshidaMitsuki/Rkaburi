# =============================================================================
# SCRIPT NAME: create_experiment_list.py
#
# PURPOSE:
#   精製済みデータセットを読み込み、リスク分析を行い、
#   Rでの再現実験に使うための「実験リスト(experiments.csv)」を生成する。
# =============================================================================
import pandas as pd
import os

# --- CONFIG ---
# 入力ファイル: 高度な再分類が完了したデータセット
INPUT_FILENAME = "R/r_final_dataset_refined.csv" 
OUTPUT_DIRECTORY = "R"
# 出力ファイル: Rスクリプトが読み込む実験リスト
OUTPUT_FILENAME = "experiments.csv" 
# 実験対象とする上位件数
NUM_EXPERIMENTS = 20
# ----------------

# --- メイン処理 ---
# ----------------

def read_first_available(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise FileNotFoundError(f"候補ファイルが見つかりません: {paths}")

if __name__ == "__main__":
    try:
        df_refined, used_path = read_first_available(INPUT_FILENAME)
        print(f"✅ データ読み込み完了: '{used_path}'")
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        print("  -> 先にデータ収集と再分類スクリプトを実行してください。")
        exit(1)

    # 1. 真の偶然(accidental)のみ抽出し、リスクスコアで上位N件
    print("\n--- ## 1. 高リスクな「真の偶然の衝突」を抽出中 ## ---")
    df_true_accidental = df_refined[df_refined['refined_type'] == 'accidental'].copy()

    if df_true_accidental.empty:
        print("「真の偶然の衝突」は見つからなかったため、実験リストは作成されませんでした。")
        exit(0)

    df_true_accidental['pushed_at_date'] = pd.to_datetime(
        df_true_accidental['pushed_at'], errors='coerce'
    ).fillna(pd.to_datetime('2000-01-01'))
    recency_score = (df_true_accidental['pushed_at_date'] - pd.to_datetime('2020-01-01T00:00:00Z')).dt.days
    df_true_accidental['risk_score'] = df_true_accidental['stars'] + recency_score * 0.1
    experiment_list = df_true_accidental.sort_values(by='risk_score', ascending=False).head(NUM_EXPERIMENTS)

    # 2. Rスクリプトが必要とする列に整形
    output_columns = ['cran_package', 'conflict_url']
    final_experiment_df = experiment_list[output_columns].copy()
    final_experiment_df['github_owner_repo'] = final_experiment_df['conflict_url'].str.replace(
        'https://github.com/', '', regex=False
    )

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    final_experiment_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ {len(final_experiment_df)}件の実験リストを '{output_path}' に保存しました。")
    print("\n--- トップ5の実験対象 ---")
    print(final_experiment_df.head(5).to_string(index=False))