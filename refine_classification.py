# =============================================================================
# SCRIPT NAME: refine_classification.py
#
# PURPOSE:
#   収集済みのデータセットを読み込み、「偶然の衝突」とされたリポジトリの
#   中身を詳しく調査し、より精度の高い分類に更新する。
# =============================================================================
import pandas as pd
import requests
import time
import os
import base64
from dotenv import load_dotenv
import re  # 正規表現モジュール
from difflib import SequenceMatcher
# --- CONFIG ---
INPUT_FILENAME = "R/r_focused_dataset.csv"
OUTPUT_DIRECTORY = "R"
OUTPUT_FILENAME = "final_dataset_refined.csv"
# ----------------

load_dotenv(os.path.join(OUTPUT_DIRECTORY, '.env'))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_cran_metadata_full(pkg_name: str, cache: dict) -> dict:
    """Metacran APIを使い、パッケージのメタデータ（Title, Description）を取得"""
    if pkg_name in cache:
        return cache[pkg_name]
    
    url = f"https://crandb.r-pkg.org/{pkg_name}"
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        metadata = {
            'Title': data.get('Title'),
            'Description': data.get('Description')
        }
        cache[pkg_name] = metadata
        time.sleep(0.2)
        return metadata
    except requests.RequestException:
        cache[pkg_name] = {}
        return {}

def get_repo_description_file(owner_repo: str) -> str:
    """GitHub APIを使い、リポジトリからDESCRIPTIONファイルの中身を取得する"""
    url = f"https://api.github.com/repos/{owner_repo}/contents/DESCRIPTION"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None # ファイルが存在しない
        
        content_b64 = response.json()['content']
        # Base64でエンコードされているのでデコードする
        decoded_content = base64.b64decode(content_b64).decode('utf-8')
        return decoded_content
    except (requests.RequestException, KeyError):
        return None

def parse_description_content(content: str) -> dict:
    """DESCRIPTIONファイルの内容（文字列）からPackage名とTitleを抽出する"""
    details = {}
    if not content:
        return details
    
    pkg_match = re.search(r"^Package:\s*(.*)", content, re.MULTILINE)
    title_match = re.search(r"^Title:\s*(.*)", content, re.MULTILINE)
    
    if pkg_match:
        details['Package'] = pkg_match.group(1).strip()
    if title_match:
        details['Title'] = title_match.group(1).strip()
        
    return details

# --- メイン処理 ---
if __name__ == "__main__":
    print(f"'{INPUT_FILENAME}' を読み込んでいます...")
    try:
        df_raw = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"❌ エラー: '{INPUT_FILENAME}' が見つかりません。")
        exit()

    print("✅ データ読み込み完了。再分類を開始します。")

    # 元のデータフレームの'accidental'行のインデックスを保持
    accidental_indices = df_raw[df_raw['conflict_type'] == 'accidental'].index
    df_accidental = df_raw.loc[accidental_indices].copy()
    
    refined_types = []
    cran_metadata_cache = {}
    
    print(f"\n--- {len(df_accidental)}件の「偶然の衝突」を詳細に調査します ---")
    
    for index, row in df_accidental.iterrows():
        cran_pkg = row['cran_package']
        conflict_url = row['conflict_url']
        description = str(row.get('description', '')).lower()
        
        current_type = 'accidental' # デフォルト
        
        # ルール1: 自己申告ミラーの判定 (より厳密に)
        if re.search(r'\bmirror\b', description): # 'mirror'を単語として検索
            current_type = 'personal_mirror'
        else:
            # ルール2: DESCRIPTIONファイルの内容が酷似するリポジトリの判定
            if conflict_url and isinstance(conflict_url, str):
                owner_repo = conflict_url.replace("https://github.com/", "")
                
                desc_content_gh = get_repo_description_file(owner_repo)
                time.sleep(1) # APIへの配慮
                
                if desc_content_gh:
                    details_gh = parse_description_content(desc_content_gh)
                    
                    if details_gh.get('Package') == cran_pkg:
                        # CRANの公式メタデータを取得 (キャッシュ利用)
                        details_cran = get_cran_metadata_full(cran_pkg, cran_metadata_cache)
                        
                        # Titleの類似度を計算 (0.0〜1.0)
                        title_gh = details_gh.get('Title', '')
                        title_cran = details_cran.get('Title', '')
                        title_similarity = SequenceMatcher(None, title_gh, title_cran).ratio()
                        
                        # Titleが90%以上似ていればクローンと判定
                        if title_similarity > 0.9:
                            current_type = 'clone_or_imitation'

        refined_types.append(current_type)
        print(f"  -> {conflict_url}  =>  {current_type}")

    # 元のデータフレームに、新しい分類結果を安全にマージ
    df_raw.loc[accidental_indices, 'refined_type'] = refined_types
    
    # 表示順を調整
    desired_column_order = [
        'cran_package', 'num_conflicts', 'conflict_type', 'refined_type', 'official_url',
        'conflict_url', 'stars', 'language', 'topics', 'created_at', 'pushed_at',
        'forks_count', 'open_issues_count', 'description', 'scraped_at_utc',
        'name_length', 'word_count', 'name_freq_score'
    ]
    final_columns = [col for col in desired_column_order if col in df_raw.columns]
    df_final = df_raw[final_columns]

    print("\n--- ## 再分類の結果サ-マリー ## ---")
    print(df_raw['refined_type'].value_counts(dropna=False))
    
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 再分類した最終結果を '{output_path}' に保存しました。")
