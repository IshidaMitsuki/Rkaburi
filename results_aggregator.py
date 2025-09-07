"""
results_aggregator.py
このスクリプトは、experiment_harness.R が保存した各実験ケースの JSON を横断的に集計し、
- 実験ケース単位のサマリー CSV
- 全体および層別の各種メトリクス JSON
- refined_type（精製タイプ）単位の集計 CSV
を生成します。

想定される各ケースの JSON 構造（experiment_harness.R の仕様に準拠）:
  {
    "cran_package": "pkgname",
    "github_repo": "owner/repo",
    "overwritten": true/false,
    "exports_diff": {"added": [...], "removed": [...]},
    "cran_version_info": {"source": "CRAN", "version": "x.y.z", ...},
    "final_version_info": {"source": "GitHub"|"Error"|..., "version": "x.y.z" | null, "error_type": "...", ...},
    // 任意項目:
    "exports": [...], "final_exports": [...]
  }

入力:
  --results_dir R/results
  --refined_csv R/final_dataset_refined.csv
  --experiments_csv R/experiments.csv

出力（デフォルトは R/summary 配下）:
  - experiments_summary.csv（各実験ケース1行）
  - metrics.json（集計指標）
  - by_refined_type.csv（refined_type 別の集計）
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# --- CONFIG (設定項目) ---
# ★★★ ここで入出力のパスを指定します ★★★
RESULTS_DIR = "R/results"
REFINED_CSV = "R/r_final_dataset_refined.csv"
OUTPUT_DIR = "R/summary"
EXPERIMENTS_CSV = "R/r_experiments.csv"

def setup_logging(verbosity: int) -> None:
    """
    ログ出力レベルの設定。
    -v を増やすほど詳細なログ（INFO, DEBUG）を出す。
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=level
    )


def list_experiment_jsons(results_dir: Path) -> List[Path]:
    """
    実験結果 JSON の一覧を取得。
    想定パス: R/results/<cran_package>/experiment_result.json
    直接存在しない場合は、念のため配下を再帰検索して拾う。
    """
    candidates = []
    if not results_dir.exists():
        logging.warning("results_dir does not exist: %s", results_dir)
        return candidates
    for pkg_dir in results_dir.iterdir():
        if not pkg_dir.is_dir():
            continue
        f = pkg_dir / "experiment_result.json"
        if f.exists():
            candidates.append(f)
        else:
            # フォールバック: 再帰的に experiment_result.json を探索
            for g in pkg_dir.rglob("experiment_result.json"):
                candidates.append(g)
    # 重複除去・安定ソート
    return sorted(set(candidates))


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    JSON ファイルを読み込んで dict を返す。失敗時は None。
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.error("Failed to read JSON %s: %s", path, e)
        return None


def safe_len(x) -> int:
    """
    list / set / tuple / dict などの要素数を安全に取得。
    None の場合は 0 を返す。
    """
    if isinstance(x, (list, set, tuple)):
        return len(x)
    if x is None:
        return 0
    if isinstance(x, dict):
        return len(x)
    return 0


def extract_version_info(info: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    version_info から source / version を取り出し、空文字は None 扱いに正規化。
    """
    if not isinstance(info, dict):
        return (None, None)
    source = info.get("source")
    version = info.get("version")
    # 空文字は None に正規化
    if isinstance(source, str) and source.strip() == "":
        source = None
    if isinstance(version, str) and version.strip() == "":
        version = None
    return (source, version)


def infer_install_success(
    final_source: Optional[str],
    final_version: Optional[str],
    overwritten: Optional[bool],
    top_level: Dict[str, Any],
) -> Tuple[Optional[bool], Optional[str]]:
    """
    GitHub 側のインストール（あるいは最終状態）成功/失敗の推定を行う。
    ヒューリスティクス（仕様の要点）:
      - final_version_info.source == "Error" は失敗
      - exports が空かつ overwritten == False の場合も失敗の可能性（ハーネスの失敗を示唆）
    可能なら top-level の install_success / install_github_success を尊重する。
    戻り値: (install_success: bool|None, error_type: str|None)
    """
    # もしハーネスが明示的な成功フラグを記録していればそれを優先
    explicit = None
    for k in ("install_success", "install_github_success"):
        if k in top_level and isinstance(top_level[k], bool):
            explicit = top_level[k]
            break

    error_type = None
    # エラーメッセージに該当しそうなフィールドを探索
    potential_error_fields = [
        ("final_version_info", "error_type"),
        ("final_version_info", "error"),
        (None, "error_type"),
        (None, "failure_reason"),
        (None, "error"),
    ]

    def get_nested(src: Dict[str, Any], outer: Optional[str], inner: str):
        if outer is None:
            return src.get(inner)
        obj = src.get(outer)
        if isinstance(obj, dict):
            return obj.get(inner)
        return None

    for outer, inner in potential_error_fields:
        v = get_nested(top_level, outer, inner)
        if isinstance(v, str) and v.strip():
            error_type = v.strip()
            break

    if explicit is not None:
        success = explicit
    else:
        if final_source == "Error":
            success = False
        else:
            # exports が空かどうか（final_exports を優先的に確認）
            exports = top_level.get("exports")
            final_exports = top_level.get("final_exports")
            exports_empty_case = False
            for ex in (final_exports, exports):
                if isinstance(ex, list):
                    if len(ex) == 0:
                        exports_empty_case = True
                        break
            # exports の明示がない場合、diff がゼロかつ overwritten=False も「空」と同様に扱う
            if not exports_empty_case:
                ediff = top_level.get("exports_diff", {})
                added = ediff.get("added", [])
                removed = ediff.get("removed", [])
                if safe_len(added) == 0 and safe_len(removed) == 0 and (overwritten is False):
                    exports_empty_case = True
            # 空かつ上書きされていないなら失敗とみなす
            if exports_empty_case and (overwritten is False):
                success = False
            else:
                # 明確な Error でなく、version もしくは source があれば成功とみなす
                success = (final_source not in (None, "Error", "")) or (bool(final_version))
    if success and not error_type:
        error_type = None
    if not success and not error_type:
        error_type = "unknown"
    return success, error_type


def flatten_result(path: Path, obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    1 件の JSON（1 実験ケース）を 1 行の辞書に平坦化。
    - exports_diff の added/removed をカウント
    - cran_version_info / final_version_info から source/version を抽出
    - ヒューリスティクスで install_github_success と error_type を推定
    """
    cran_package = obj.get("cran_package") or path.parent.name
    github_repo = obj.get("github_repo") or obj.get("github_owner_repo") or None

    overwritten = obj.get("overwritten")
    if not isinstance(overwritten, bool):
        overwritten = None

    exports_diff = obj.get("exports_diff") or {}
    added = exports_diff.get("added", [])
    removed = exports_diff.get("removed", [])
    added_count = safe_len(added)
    removed_count = safe_len(removed)

    cran_source, cran_version = extract_version_info(obj.get("cran_version_info"))
    final_source, final_version = extract_version_info(obj.get("final_version_info"))

    install_success, error_type = infer_install_success(final_source, final_version, overwritten, obj)

    row = {
        "cran_package": cran_package,
        "github_repo": github_repo,
        "overwritten": overwritten,
        "added_count": added_count,
        "removed_count": removed_count,
        "cran_source": cran_source,
        "final_source": final_source,
        "cran_version": cran_version,
        "final_version": final_version,
        "install_github_success": install_success,
        "error_type": error_type,
        # デバッグ用に元 JSON のパスも保持
        "_json_path": str(path),
    }
    return row


def compute_iqr(series: pd.Series) -> Optional[float]:
    """
    IQR（四分位範囲）を計算する補助関数。
    現状このスクリプトでは未使用だが、将来の拡張を見越して残している。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return float(q3 - q1)


def smart_merge(base: pd.DataFrame, other: pd.DataFrame, key_candidates: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    DataFrame を「使える最初のキーの組み合わせ」で左外部結合する。
    key_candidates は (左のキー, 右のキー) の優先順リスト。
    最初に両方の列が存在した組み合わせで merge を実行する。
    """
    for left_key, right_key in key_candidates:
        if left_key in base.columns and right_key in other.columns:
            logging.info("Merging on %s <-> %s", left_key, right_key)
            return base.merge(other, how="left", left_on=left_key, right_on=right_key)
    logging.warning("No matching keys found for merge. Returning base unchanged.")
    return base


def load_optional_csv(path: Optional[Path], label: str) -> Optional[pd.DataFrame]:
    """
    CSV の存在を確認してから読み込む。失敗時は None。
    """
    if not path:
        return None
    if not path.exists():
        logging.warning("%s not found: %s", label, path)
        return None
    try:
        df = pd.read_csv(path)
        logging.info("Loaded %s rows from %s", len(df), path)
        return df
    except Exception as e:
        logging.error("Failed to read %s at %s: %s", label, path, e)
        return None


def build_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    集計指標（メトリクス）を算出して dict で返す。
    - 全体件数
    - 上書き率（全体）
    - インストール成功率（全体）
    - added/removed の中央値・IQR・最大
    - 失敗理由の分布
    - refined_type / final_source 別の上書き率・成功率
    """
    metrics: Dict[str, Any] = {}
    n = len(df)
    metrics["n_cases"] = int(n)

    # 上書き率（全体）
    if "overwritten" in df.columns:
        over = pd.to_numeric(df["overwritten"].astype(float), errors="coerce")
        metrics["overwrite_rate_overall"] = float(over.mean()) if len(over) else None

    # インストール成功率（全体）
    if "install_github_success" in df.columns:
        suc = pd.to_numeric(df["install_github_success"].astype(float), errors="coerce")
        metrics["install_success_rate_overall"] = float(suc.mean()) if len(suc) else None

    # added / removed の要約統計
    def stats_for(col: str) -> Dict[str, Any]:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        if len(s) == 0:
            return {"median": None, "iqr": None, "max": None}
        return {
            "median": float(s.median()),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "max": float(s.max()),
        }

    metrics["added_count_stats"] = stats_for("added_count")
    metrics["removed_count_stats"] = stats_for("removed_count")

    # 失敗ケースの error_type 分布
    if "install_github_success" in df.columns:
        failures = df[df["install_github_success"] == False]
        if "error_type" in failures.columns:
            vc = failures["error_type"].fillna("unknown").astype(str).value_counts(dropna=False)
            metrics["failure_reason_distribution"] = {str(k): int(v) for k, v in vc.items()}

    # refined_type 別の上書き率 / 成功率
    if "refined_type" in df.columns:
        grp = df.groupby("refined_type")["overwritten"].mean(numeric_only=True)
        metrics["overwrite_rate_by_refined_type"] = {str(k): (float(v) if not math.isnan(v) else None) for k, v in grp.items()}
        grp_s = df.groupby("refined_type")["install_github_success"].mean(numeric_only=True)
        metrics["install_success_rate_by_refined_type"] = {str(k): (float(v) if not math.isnan(v) else None) for k, v in grp_s.items()}

    # final_source 別の上書き率
    if "final_source" in df.columns and "overwritten" in df.columns:
        grp2 = df.groupby("final_source")["overwritten"].mean(numeric_only=True)
        metrics["overwrite_rate_by_final_source"] = {str(k): (float(v) if not math.isnan(v) else None) for k, v in grp2.items()}

    return metrics


def aggregate_by_refined_type(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    refined_type 単位での集計 DataFrame を作成。
    件数、上書き率、成功率、added/removed の中央値・IQR・最大を計算。
    """
    if "refined_type" not in df.columns:
        return None
    out = []
    for rtype, g in df.groupby("refined_type", dropna=False):
        n = len(g)
        overw = pd.to_numeric(g["overwritten"].astype(float), errors="coerce")
        succ = pd.to_numeric(g["install_github_success"].astype(float), errors="coerce")

        added = pd.to_numeric(g["added_count"], errors="coerce")
        removed = pd.to_numeric(g["removed_count"], errors="coerce")

        def med(series): return float(series.median()) if series.notna().any() else None
        def iqr(series):
            s = series.dropna()
            return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else None
        def mx(series): return float(series.max()) if series.notna().any() else None

        out.append({
            "refined_type": rtype,
            "n": int(n),
            "overwrite_rate": float(overw.mean()) if overw.notna().any() else None,
            "install_success_rate": float(succ.mean()) if succ.notna().any() else None,
            "added_median": med(added),
            "added_iqr": iqr(added),
            "added_max": mx(added),
            "removed_median": med(removed),
            "removed_iqr": iqr(removed),
            "removed_max": mx(removed),
        })
    return pd.DataFrame(out)


def safe_drop_duplicates(df: pd.DataFrame, preferred_columns: List[str], label: str) -> pd.DataFrame:
    """
    指定された列のうち、存在する最初の列で重複削除を実行
    """
    if df is None:
        return df
    
    for col in preferred_columns:
        if col in df.columns:
            logging.info(f"{label}: {col}列で重複削除を実行")
            return df.drop_duplicates(subset=[col])
    
    logging.warning(f"{label}: 重複削除用の列が見つかりません。利用可能な列: {list(df.columns)}")
    return df


def main():
    """
    エントリポイント:
    - CONFIGで定義されたパスを使用
    - JSONの読み込みと平坦化
    - 補助CSVの読み込みとマージ
    - サマリー成果物の出力
    """
    setup_logging(1)
    
    # argparseの部分を削除し、CONFIG変数を直接Pathオブジェクトに変換
    results_dir = Path(RESULTS_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    refined_path = Path(REFINED_CSV) if REFINED_CSV else None
    experiments_path = Path(EXPERIMENTS_CSV) if EXPERIMENTS_CSV else None

    # 実験結果JSON一覧の収集
    json_paths = list_experiment_jsons(results_dir)
    logging.info("%d 件の experiment_result.json ファイルを発見しました", len(json_paths))

    # JSONを読み込み、1行のdictに平坦化
    rows: List[Dict[str, Any]] = []
    for p in json_paths:
        obj = read_json(p)
        if obj is None:
            continue
        row = flatten_result(p, obj)
        rows.append(row)

    # 1件も読めなかった場合も、空の出力ファイルを作って終了
    if not rows:
        logging.error("有効な実験結果JSONが見つかりませんでした。")
        return

    summary_df = pd.DataFrame(rows)

    # 補助CSVの読み込み
    refined_df = load_optional_csv(refined_path, "精製済みCSV")
    experiments_df = load_optional_csv(experiments_path, "実験リストCSV")

    # ★★★ 修正2: 補助CSVの正規化と重複除去（安全に） ★★★
    if experiments_df is not None:
        # github_owner_repo が無ければ conflict_url から作成
        if "github_owner_repo" not in experiments_df.columns and "conflict_url" in experiments_df.columns:
            experiments_df["github_owner_repo"] = experiments_df["conflict_url"].astype(str).str.replace("https://github.com/", "", regex=False)
        if "github_owner_repo" in experiments_df.columns:
            experiments_df = experiments_df.drop_duplicates(subset=["github_owner_repo"])

    if refined_df is not None:
        # refined 側は github_repo を conflict_url から作成（無い場合のみ）
        if "github_repo" not in refined_df.columns and "conflict_url" in refined_df.columns:
            refined_df["github_repo"] = refined_df["conflict_url"].astype(str).str.replace("https://github.com/", "", regex=False)
        if "github_repo" in refined_df.columns:
            refined_df = refined_df.drop_duplicates(subset=["github_repo"])

    # ★★★ 修正3: experiments.csvとマージ（repo優先） ★★★
    if experiments_df is not None:
        key_pairs_exp = [
            ("github_repo", "github_owner_repo"),  # repo優先
            ("cran_package", "cran_package"),
        ]
        summary_df = smart_merge(summary_df, experiments_df, key_pairs_exp)
        
        # 空の github_repo を github_owner_repo で補完（両列がある場合）
        if "github_owner_repo" in summary_df.columns and "github_repo" in summary_df.columns:
            summary_df["github_repo"] = summary_df["github_repo"].fillna(summary_df["github_owner_repo"])

    # ★★★ 修正3: refined_csvとマージ（repo優先） ★★★
    if refined_df is not None:
        key_pairs_refined = [
            ("github_repo", "github_repo"),  # repo優先
            ("cran_package", "cran_package"),
        ]
        summary_df = smart_merge(summary_df, refined_df, key_pairs_refined)

    # 出力カラムの希望順
    desired_cols = [
        "cran_package", "github_repo", "overwritten", "added_count", "removed_count",
        "cran_source", "final_source", "cran_version", "final_version",
        "install_github_success", "error_type", "refined_type", "stars",
        "pushed_at", "risk_score",
    ]
    out_cols = [c for c in desired_cols if c in summary_df.columns]
    
    # ★★★ 修正4: 出力直前の安全な重複除去 ★★★
    dedup_keys = [c for c in ["cran_package", "github_repo", "cran_version", "final_version", "added_count", "removed_count"] if c in summary_df.columns]
    if dedup_keys:
        before = len(summary_df)
        summary_df = summary_df.drop_duplicates(subset=dedup_keys)
        logging.info("Dropped %d duplicate rows based on %s", before - len(summary_df), dedup_keys)
    
    # boolカラムを正規化
    for bcol in ("overwritten", "install_github_success"):
        if bcol in summary_df.columns:
            summary_df[bcol] = summary_df[bcol].map(lambda x: bool(x) if pd.notna(x) else np.nan)

    # experiments_summary.csvの書き出し
    summary_path = out_dir / "experiments_summary.csv"
    summary_df[out_cols].to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info("詳細な実験結果サマリーを '%s' に保存しました。", summary_path)

    # metrics.jsonの作成・書き出し
    metrics = build_metrics(summary_df)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    logging.info("主要な集計指標を '%s' に保存しました。", metrics_path)

    # by_refined_type.csvの出力
    by_refined = aggregate_by_refined_type(summary_df)
    by_refined_path = out_dir / "by_refined_type.csv"
    if by_refined is not None and not by_refined.empty:
        by_refined.to_csv(by_refined_path, index=False, encoding="utf-8-sig")
        logging.info("カテゴリ別集計を '%s' に保存しました。", by_refined_path)
    else:
        by_refined_path.write_text("", encoding="utf-8")
        logging.info("refined_type列がないため、空のファイルを出力しました: %s", by_refined_path)

if __name__ == "__main__":
    main()