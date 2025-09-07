# =============================================================================
# SCRIPT NAME: experiment_harness.R
# PURPOSE:
#   'experiments.csv' を読み込み、各衝突事例を隔離環境で再現実験。
#   上書き挙動・エクスポート差分・メタ情報を保存し、サマリーCSVも出力。
# =============================================================================

# --- 必要なパッケージ ---
if (!requireNamespace("withr", quietly = TRUE)) install.packages("withr")
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

library(withr)
library(remotes)
library(jsonlite)

# --- 実験の安定化設定 ---
options(timeout = 600)  # ネットワーク待ちを延長
Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS = "true")  # remotesの警告をエラー化しない

# --- 設定 ---
INPUT_FILE <- "R/r_experiments.csv"
OUTPUT_DIR <- "R/results"

# --- ヘルパー関数 ---
get_package_info <- function(pkg_name, lib_path) {
  tryCatch({
    desc <- utils::packageDescription(pkg_name, lib.loc = lib_path)
    
    # インストール元の推定ロジック（優先順: RemoteType -> GithubRepo -> Repository）
    source <- "Unknown"
    if (!is.null(desc$RemoteType) && grepl("github", desc$RemoteType, ignore.case = TRUE)) {
      source <- "GitHub"
    } else if (!is.null(desc$GithubRepo)) {
      source <- "GitHub"
    } else if (!is.null(desc$Repository) && grepl("CRAN", desc$Repository, ignore.case = TRUE)) {
      source <- "CRAN"
    } else if (!is.null(desc$Repository)) {
      source <- desc$Repository
    }
    
    remote_user <- if (!is.null(desc$RemoteUsername)) desc$RemoteUsername else NA_character_
    remote_repo <- if (!is.null(desc$RemoteRepo))      desc$RemoteRepo      else NA_character_
    remote_sha  <- if (!is.null(desc$RemoteSha))       desc$RemoteSha       else NA_character_
    
    exports <- tryCatch(getNamespaceExports(pkg_name), error = function(e) character(0))
    
    list(
      version = if (!is.null(desc$Version)) desc$Version else NA_character_,
      source  = source,
      repository = if (!is.null(desc$Repository)) desc$Repository else NA_character_,
      remote = list(
        owner_repo = if (!is.na(remote_user) && !is.na(remote_repo)) paste0(remote_user, "/", remote_repo) else NA_character_,
        sha = remote_sha
      ),
      exports = exports
    )
  }, error = function(e) {
    list(version = NA_character_, source = "Error", repository = NA_character_, remote = list(owner_repo = NA_character_, sha = NA_character_), exports = character(0))
  })
}

# --- メイン処理 ---
if (!file.exists(INPUT_FILE)) {
  stop("Error: '", INPUT_FILE, "' が見つかりません。先に Python スクリプトを実行してください。")
}

experiments <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

message(paste("---", nrow(experiments), "件の再現実験を開始します ---"))
summary_rows <- list()

for (i in seq_len(nrow(experiments))) {
  cran_pkg    <- experiments$cran_package[i]
  github_repo <- experiments$github_owner_repo[i]
  
  message(paste0("\n[", i, "/", nrow(experiments), "] Processing: ", cran_pkg, " vs ", github_repo))
  
  with_temp_libpaths({
    # 1. CRAN版インストール
    message("  -> 1. CRAN 版をインストール中...")
    tryCatch(
      install.packages(cran_pkg, repos = "https://cran.rstudio.com/", quiet = TRUE, dependencies = TRUE),
      error = function(e) message("     !! CRAN install error: ", conditionMessage(e))
    )
    info_after_cran <- get_package_info(cran_pkg, .libPaths()[1])
    
    # 2. GitHub版インストール
    message("  -> 2. GitHub版をインストール中...")
    tryCatch(
      remotes::install_github(github_repo, quiet = TRUE, force = TRUE, dependencies = TRUE, upgrade = "never"),
      error = function(e) message("     !! GitHub install error: ", conditionMessage(e))
    )
    info_after_github <- get_package_info(cran_pkg, .libPaths()[1])
    
    # 3. 差分と上書き判定
    added_exports   <- setdiff(info_after_github$exports, info_after_cran$exports)
    removed_exports <- setdiff(info_after_cran$exports, info_after_github$exports)
    overwritten <- isTRUE(info_after_cran$source != info_after_github$source)
    
    result <- list(
      cran_package = cran_pkg,
      github_repo  = github_repo,
      cran_version_info  = info_after_cran,
      final_version_info = info_after_github,
      overwritten  = overwritten,
      exports_diff = list(
        added = added_exports,
        removed = removed_exports
      )
    )
    
    # 4. ケースごとのJSON保存
    case_output_dir <- file.path(OUTPUT_DIR, cran_pkg, gsub("/", "__", github_repo))
    dir.create(case_output_dir, showWarnings = FALSE, recursive = TRUE)
    jsonlite::write_json(result, file.path(case_output_dir, "experiment_result.json"), auto_unbox = TRUE, pretty = TRUE)
    message("  ->    実験完了。結果を保存しました。")
    
    # 5. サマリー行（CSV用）
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      cran_package           = cran_pkg,
      github_repo            = github_repo,
      overwritten            = overwritten,
      cran_version           = info_after_cran$version,
      final_version          = info_after_github$version,
      source_before          = info_after_cran$source,
      source_after           = info_after_github$source,
      added_exports_count    = length(added_exports),
      removed_exports_count  = length(removed_exports),
      final_remote_ownerrepo = if (!is.null(info_after_github$remote$owner_repo)) info_after_github$remote$owner_repo else NA_character_,
      final_remote_sha       = if (!is.null(info_after_github$remote$sha)) info_after_github$remote$sha else NA_character_,
      stringsAsFactors = FALSE
    )
  }) # end with_temp_libpaths
  
  # GitHubへの負荷対策
  Sys.sleep(2)
}

# 全体サマリーCSV出力
if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  utils::write.csv(summary_df, file = file.path(OUTPUT_DIR, "summary.csv"), row.names = FALSE, fileEncoding = "UTF-8")
}

message("\n--- 全ての実験が完了しました ---")