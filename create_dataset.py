# =============================================================================
# SCRIPT NAME: create_final_dataset_v7_complete.py
#
# PURPOSE:
#   - これまでの議論の全ての機能（件数調整、キャッシュ、名前スコア、
#     衝突なし記録、全特徴量）を搭載した。
# =============================================================================

import requests
import pandas as pd
import time
import os
import re
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from difflib import SequenceMatcher
import base64
import math
# --- CONFIG ---
NUMBER_OF_PACKAGES_TO_ANALYZE = 200
MAX_GITHUB_REPOS_PER_PACKAGE = 200
OUTPUT_DIRECTORY = "R"
CACHE_DIRECTORY = "cache"
OUTPUT_FILENAME = "final_dataset_for_analysis.csv"
# ----------------

# .envファイルを読み込む
load_dotenv(os.path.join(OUTPUT_DIRECTORY, '.env'))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_top_packages(count: int) -> list:
    """CRANの人気パッケージリストを取得する"""
    print(f"CRANで人気のトップ{count}パッケージを取得します...")
    try:
        url = f"https://cranlogs.r-pkg.org/top/last-month/{count}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'downloads' in data:
            return [item['package'] for item in data['downloads']]
    except Exception as e:
        print(f"❌ APIでのパッケージリスト取得に失敗: {e}")
        return []

def create_word_freq_scorer():
    """英語の単語頻度リストを読み込み、パッケージ名のスコアを計算する関数を返す"""
    print("名前の一般性を評価するための、英単語頻度リストを準備中...")
    try:
        url = "https://norvig.com/ngrams/count_1w.txt"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        lines = response.text.splitlines()
        total_words = sum(int(line.split('\t')[1]) for line in lines)
        freq_dict = {line.split('\t')[0]: int(line.split('\t')[1]) / total_words for line in lines}
        
        def calculate_score(pkg_name):
            words = re.split(r'[._-]', pkg_name.lower())
            log_prob_sum = sum(math.log(freq_dict.get(w, 1e-9)) for w in words)
            return log_prob_sum
        print("✅ 単語頻度スコアの準備が完了しました。")
        return calculate_score
    except Exception as e:
        print(f"❌ 単語頻度リストの取得に失敗: {e}。スコア計算をスキップします。")
        return lambda pkg_name: None

def get_cran_metadata(pkg_name: str, cache: dict) -> dict:
    """Metacran APIを使い、公式URLやMaintainerなど詳細メタデータを取得"""
    if pkg_name in cache:
        return cache[pkg_name]
    
    url = f"https://crandb.r-pkg.org/{pkg_name}"
    metadata = {'url': None, 'maintainer': None, 'title': None}
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        # 公式URLの抽出
        urls = data.get('URL', '')
        bug_reports = data.get('BugReports', '')
        combined_text = str(urls) + " " + str(bug_reports)
        match = re.search(r'(https://github.com/([^/]+)/[^/,\s]+)', combined_text)
        if match:
            metadata['url'] = match.group(1).strip('/')
        
        # Maintainer名の抽出
        maintainer_info = data.get('Maintainer', '')
        maintainer_match = re.search(r'<(.*)>', maintainer_info) # メールアドレス部分から推測
        if maintainer_match:
            metadata['maintainer'] = maintainer_match.group(1).split('@')[0].lower()

        metadata['title'] = data.get('Title')
        cache[pkg_name] = metadata
        return metadata
    except requests.RequestException:
        cache[pkg_name] = metadata
        return metadata

def get_repo_description_content(owner_repo: str) -> str:
    """GitHub APIを使い、リポジトリからDESCRIPTIONファイルの中身を取得する"""
    url = f"https://api.github.com/repos/{owner_repo}/contents/DESCRIPTION"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None
        content_b64 = response.json()['content']
        return base64.b64decode(content_b64).decode('utf-8')
    except:
        return None

def parse_description_content(content: str) -> dict:
    """DESCRIPTIONファイルの内容からPackage名とTitleを抽出する"""
    details = {}
    if not content: return details
    pkg_match = re.search(r"^Package:\s*(.*)", content, re.MULTILINE)
    title_match = re.search(r"^Title:\s*(.*)", content, re.MULTILINE)
    if pkg_match: details['Package'] = pkg_match.group(1).strip()
    if title_match: details['Title'] = title_match.group(1).strip()
    return details

def search_github_repos(pkg_name: str, max_results: int) -> list:
    """GitHub APIを使い、リポジトリを検索 (改良版キャッシュ・件数制限機能付き)"""
    cache_file = os.path.join(CACHE_DIRECTORY, f"{pkg_name}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_repos_in_cache = json.load(f)
        if len(all_repos_in_cache) >= max_results:
            return all_repos_in_cache[:max_results]

    repos = []
    search_name = pkg_name.lower()
    url = f"https://api.github.com/search/repositories?q={search_name}+in:name"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    
    while url:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            for item in data.get('items', []):
                if item['name'].lower() == search_name:
                    repos.append({
                        'url': item['html_url'].strip('/'), 'is_fork': item['fork'],
                        'description': item.get('description', ''), 'stars': item.get('stargazers_count', 0),
                        'language': item.get('language', ''), 'topics': ", ".join(item.get('topics', [])),
                        'created_at': item.get('created_at', ''), 'pushed_at': item.get('pushed_at', ''),
                        'forks_count': item.get('forks_count', 0), 'open_issues_count': item.get('open_issues_count', 0)
                    })
                if len(repos) >= max_results: break
            if len(repos) >= max_results: break
            url = response.links.get('next', {}).get('url')
            time.sleep(2)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"... 403 Forbidden for {pkg_name}: APIレート制限に達しました。60秒待機します ...")
                time.sleep(60)
                continue
            else:
                 print(f"❌ HTTPエラー ({pkg_name}): {e}")
                 break
        except requests.RequestException as e:
            print(f"❌ GitHub API検索エラー ({pkg_name}): {e}")
            break
            
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(repos, f, ensure_ascii=False, indent=2)
    return repos[:max_results]

# --- メイン処理 ---
if __name__ == "__main__":
    if not GITHUB_TOKEN or GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print("❌ エラー: GITHUB_TOKEN が設定されていません。コード冒頭の変数を編集するか、.envファイルを正しく設定してください。")
    else:
        name_scorer = create_word_freq_scorer()
        top_pkgs = get_top_packages(count=NUMBER_OF_PACKAGES_TO_ANALYZE)
        results = []
        cran_meta_cache = {}

        print("\n--- CRANパッケージとGitHubリポジトリの照合を開始 ---")
        for i, pkg in enumerate(top_pkgs):
            print(f"Processing ({i+1}/{len(top_pkgs)}): {pkg}")
            
            # 事前にCRANのメタ情報を取得（キャッシュ利用）
            cran_meta = get_cran_metadata(pkg, cran_meta_cache)
            official_url = cran_meta.get('url')
            
            # GitHubリポジトリを検索
            github_repos = search_github_repos(pkg, max_results=MAX_GITHUB_REPOS_PER_PACKAGE)
            
            # パッケージ名の特徴量を計算
            name_freq_score = name_scorer(pkg) if name_scorer else None
            name_length = len(pkg)
            word_count = len(re.split(r'[._-]', pkg))

            # 衝突なしのケース
            if not github_repos:
                results.append({
                    'cran_package': pkg,
                    'name_length': name_length,
                    'word_count': word_count,
                    'name_freq_score': name_freq_score,
                    'num_conflicts': 0,
                    'conflict_type': 'none',
                    'is_same_owner': None,
                    'title_similarity_score': 0.0,
                    'official_url': official_url,
                    'scraped_at_utc': datetime.now(timezone.utc).isoformat(),
                    'conflict_url': None, 'stars': 0, 'language': None, 'topics': None,
                    'created_at': None, 'pushed_at': None, 'forks_count': 0,
                    'open_issues_count': 0, 'description': None
                })
                continue

            # 衝突があった場合の処理
            num_conflicts = len(github_repos)
            for repo in github_repos:
                repo_owner = repo['url'].split('/')[-2].lower()
                is_same_owner = (cran_meta.get('maintainer') is not None and cran_meta.get('maintainer') in repo_owner)
                
                repo_type = ""
                title_similarity_score = 0.0
                
                if official_url and repo['url'].lower() == official_url.lower():
                    repo_type = "official"
                elif repo['is_fork']:
                    repo_type = "fork"
                else:
                    repo_type = "accidental" # まずは仮判定
                    owner_repo_str = repo['url'].replace("https://github.com/", "")
                    desc_content = get_repo_description_content(owner_repo_str)
                    time.sleep(0.5)
                    if desc_content:
                        gh_desc_details = parse_description_content(desc_content)
                        if gh_desc_details.get('Package') == pkg:
                            title_gh = gh_desc_details.get('Title', '')
                            title_cran = cran_meta.get('title', '')
                            if title_gh and title_cran:
                                title_similarity_score = SequenceMatcher(None, title_gh, title_cran).ratio()
                                if title_similarity_score > 0.9:
                                    repo_type = "clone_or_imitation"
                
                results.append({
                    'cran_package': pkg,
                    'name_length': name_length,
                    'word_count': word_count,
                    'name_freq_score': name_freq_score,
                    'num_conflicts': num_conflicts,
                    'conflict_type': repo_type,
                    'is_same_owner': is_same_owner,
                    'title_similarity_score': title_similarity_score,
                    'official_url': official_url,
                    'scraped_at_utc': datetime.now(timezone.utc).isoformat(),
                    'conflict_url': repo['url'],
                    'stars': repo['stars'],
                    'language': repo['language'],
                    'topics': repo['topics'],
                    'created_at': repo['created_at'],
                    'pushed_at': repo['pushed_at'],
                    'forks_count': repo['forks_count'],
                    'open_issues_count': repo['open_issues_count'],
                    'description': repo['description']
                })
        
        df = pd.DataFrame(results)
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 全ての処理が完了しました。最終データセットを '{output_path}' に保存しました。")