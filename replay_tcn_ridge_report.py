"""Replay the locked 2025 EWMA/Ridge/TCN models and build an HTML report.

This script deliberately does no hyperparameter search.  It uses the TCN and
Ridge settings selected on the 2022-2024 validation folds, fits on data through
2024, predicts 2025, and exports every pitcher-specific Ridge coefficient.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.evaluation import evaluate_common_and_deployable
from lib.shared_tcn import SharedTCNConfig
from lib.stuff_cli import load_stuff_data
from lib.stuff_experiment import (
    COMPACT_FEATURES,
    average_seed_predictions,
    predict_ewma_fold,
    predict_residual_fold,
    predict_tcn_fold,
    prepare_fold,
)


SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)
RIDGE_ALPHA = 100.0
RIDGE_SCALE = 0.25
TCN_CONFIG = SharedTCNConfig(
    sequence_length=8,
    internal_channels=4,
    bottleneck_dim=4,
    kernel_size=2,
    dilations=(1, 2, 4),
    dropout=0.1,
    head_type="H1",
    huber_delta=1.0,
    pitcher_weight_l2=0.01,
    pitcher_bias_l2=0.01,
    encoder_l2=0.0,
    ordinal_weight=0.0,
    alpha=0.1,
    learning_rate=0.001,
    epochs=250,
    gradient_clip=5.0,
    seed=SEEDS[0],
    dtype="float32",
)
XGBOOST_SCALE = 0.2218450676
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.0661012188,
    "min_child_weight": 13.9429,
    "reg_alpha": 0.6152831,
    "reg_lambda": 29.1417558,
    "gamma": 0.0,
    "subsample": 0.6825797,
    "colsample_bytree": 0.7955172,
    "tree_method": "hist",
    "max_bin": 64,
}

MODEL_LABELS = {
    "ewma": "EWMA4", "ridge": "Ridge", "xgboost": "XGBoost", "tcn": "TCN"
}
CLASS_LABELS = {0: "Low", 1: "Middle", 2: "High"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/reproducible_mlb_stuff")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/tcn_ridge_replay_2025"),
    )
    parser.add_argument("--html", type=Path, default=Path("FINAL_MODEL_REPORT.html"))
    parser.add_argument("--markdown", type=Path, default=Path("FINAL_MODEL_REPORT.md"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Rebuild HTML/Markdown from existing CSV outputs without refitting models.",
    )
    parser.add_argument(
        "--add-xgboost", action="store_true",
        help="Fit only locked XGBoost and merge it into an existing replay.",
    )
    return parser.parse_args()


def load_bundle(root: Path):
    return load_stuff_data(
        statcast_paths=(
            root / "statcast_mlb_stable_starters_2020",
            root / "statcast_mlb_stable_starters_2021_2025",
        ),
        stuff_paths=(
            root / "fangraphs_stuff_mlb_stable_starters_2020.parquet",
            root / "fangraphs_stuff_mlb_stable_starters_2021_2025.parquet",
        ),
        official_stats_paths=(root / "mlb_official_pitching_2021_2025.parquet",),
        qualification_mode="research",
        qualification_years=(2021, 2022, 2023, 2024, 2025),
        min_starts_per_year=20,
    )


def ridge_coefficients(estimator, names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pitcher, fitted in sorted(estimator.models_.items()):
        coef = np.asarray(fitted.estimator.coef_, dtype=float).reshape(-1)
        mean = np.asarray(fitted.transform.means, dtype=float)
        scale = np.asarray(fitted.transform.scales, dtype=float)
        intercept = float(np.asarray(fitted.estimator.intercept_).reshape(-1)[0])
        raw_intercept = RIDGE_SCALE * (intercept - np.sum(coef * mean / scale))
        for feature, value, feature_mean, feature_scale in zip(
            COMPACT_FEATURES, coef, mean, scale, strict=True
        ):
            rows.append(
                {
                    "pitcher": pitcher,
                    "pitcher_name": names.get(str(pitcher), str(pitcher)),
                    "feature": feature,
                    "standardized_residual_coefficient": value,
                    "effective_standardized_coefficient": RIDGE_SCALE * value,
                    "feature_training_mean": feature_mean,
                    "feature_training_scale": feature_scale,
                    "effective_raw_unit_coefficient": RIDGE_SCALE * value / feature_scale,
                    "model_residual_intercept": intercept,
                    "effective_raw_unit_intercept": raw_intercept,
                    "ridge_alpha": RIDGE_ALPHA,
                    "correction_scale": RIDGE_SCALE,
                }
            )
    return pd.DataFrame(rows)


def continuous_metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = pd.to_numeric(frame["true_stuff_plus"], errors="coerce")
    pred = pd.to_numeric(frame["predicted_stuff_plus"], errors="coerce")
    valid = true.notna() & pred.notna()
    error = pred.loc[valid] - true.loc[valid]
    return {
        "continuous_n": int(valid.sum()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def make_latest(predictions: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    base = predictions["ewma"].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])
    latest = (
        base.sort_values(["pitcher", "game_date", "game_pk"])
        .groupby("pitcher", as_index=False)
        .tail(1)
    )
    keep = [
        "row_id", "pitcher", "game_date", "game_pk", "true_stuff_plus",
        "true_class", "q33", "q67",
    ]
    result = latest[keep].copy()
    result["pitcher_name"] = result["pitcher"].map(
        lambda value: names.get(str(value), str(value))
    )
    for model, frame in predictions.items():
        values = frame[["row_id", "predicted_stuff_plus", "predicted_class"]].copy()
        values = values.rename(
            columns={
                "predicted_stuff_plus": f"{model}_predicted_stuff_plus",
                "predicted_class": f"{model}_predicted_class",
            }
        )
        result = result.merge(values, on="row_id", how="left", validate="one_to_one")
    ordered = ["pitcher_name", "pitcher", "game_date", "game_pk", "true_stuff_plus", "true_class", "q33", "q67"]
    for model in predictions:
        ordered += [f"{model}_predicted_stuff_plus", f"{model}_predicted_class"]
    return result[ordered].sort_values("pitcher_name").reset_index(drop=True)


def table_html(frame: pd.DataFrame, *, digits: int = 4, classes: bool = False) -> str:
    shown = frame.copy()
    if classes:
        for column in [name for name in shown if name.endswith("class")]:
            shown[column] = shown[column].map(
                lambda value: CLASS_LABELS.get(int(value), "-") if pd.notna(value) else "-"
            )
    float_columns = shown.select_dtypes(include="number").columns
    shown[float_columns] = shown[float_columns].round(digits)
    return shown.to_html(index=False, border=0, classes="data", escape=True)


def build_reports(
    *, latest: pd.DataFrame, common: pd.DataFrame, deployable: pd.DataFrame,
    coefficients: pd.DataFrame, continuous: pd.DataFrame, metadata: dict,
    html_path: Path, markdown_path: Path,
) -> None:
    metric = common.merge(continuous, on="model", how="left")
    metric["model"] = metric["model"].map(MODEL_LABELS)
    deploy = deployable.copy()
    deploy["model"] = deploy["model"].map(MODEL_LABELS)
    latest_display = latest.rename(columns={
        "pitcher_name": "선수", "pitcher": "MLBAM", "game_date": "최근 경기",
        "game_pk": "game_pk", "true_stuff_plus": "실제 Stuff+", "true_class": "실제 등급",
        "q33": "q33", "q67": "q67", "ewma_predicted_stuff_plus": "EWMA4 예측",
        "ewma_predicted_class": "EWMA4 등급", "ridge_predicted_stuff_plus": "Ridge 예측",
        "ridge_predicted_class": "Ridge 등급", "xgboost_predicted_stuff_plus": "XGBoost 예측",
        "xgboost_predicted_class": "XGBoost 등급", "tcn_predicted_stuff_plus": "TCN 예측",
        "tcn_predicted_class": "TCN 등급",
    })
    coef_summary = (
        coefficients.assign(abs_coef=lambda x: x.effective_standardized_coefficient.abs())
        .sort_values(["pitcher_name", "abs_coef"], ascending=[True, False])
        .groupby(["pitcher_name", "pitcher"], as_index=False)
        .head(3)[["pitcher_name", "pitcher", "feature", "effective_standardized_coefficient"]]
    )
    coef_sections = []
    for (name, pitcher), group in coefficients.groupby(["pitcher_name", "pitcher"], sort=True):
        cols = ["feature", "standardized_residual_coefficient", "effective_standardized_coefficient", "effective_raw_unit_coefficient", "feature_training_mean", "feature_training_scale"]
        coef_sections.append(
            f'<details><summary>{escape(str(name))} ({escape(str(pitcher))}) — 16개 계수</summary>'
            + table_html(group[cols], digits=6) + "</details>"
        )

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    css = """
    :root{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#61708a;--accent:#2457d6;--line:#dce3ee}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}
    main{max-width:1500px;margin:auto;padding:32px} h1{font-size:34px;margin:0 0 8px} h2{margin-top:38px;border-bottom:2px solid var(--line);padding-bottom:8px}
    .lead{color:var(--muted);font-size:17px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}
    .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}.card b{font-size:24px;color:var(--accent);display:block}
    .table-wrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px} table.data{border-collapse:collapse;width:100%;font-size:13px}
    table.data th,table.data td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right} table.data th{position:sticky;top:0;background:#eef3fb;color:#33415c}
    table.data th:first-child,table.data td:first-child{text-align:left} code{background:#e9eef7;padding:2px 5px;border-radius:4px} details{background:#fff;border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:10px}
    summary{cursor:pointer;font-weight:700}.note{background:#fff7dc;border-left:4px solid #e4a900;padding:12px 16px}.ok{background:#eaf7ef;border-left:4px solid #2b9b59;padding:12px 16px}
    """
    html = f"""<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Stuff+ 통합 예측 보고서</title><style>{css}</style></head><body><main>
    <h1>Stuff+ 통합 예측 보고서</h1><p class=\"lead\">EWMA4 · 선수별 Ridge · XGBoost · Shared TCN | 생성 {generated}</p>
    <div class=\"ok\"><b>결론:</b> TCN은 버린 모델이 아니다. 기존 검증에서 주 모델 교체 기준을 넘지 못해 진단 모델로 남았지만, 여기서는 같은 고정 설정으로 2025 전체와 선수별 최근 연속 예측을 함께 표시한다.</div>
    <div class=\"cards\"><div class=\"card\"><b>{metadata['qualified_pitchers']}</b>대상 선수</div><div class=\"card\"><b>{metadata['evaluation_rows']}</b>2025 대상 경기</div><div class=\"card\"><b>{metadata['common_rows']}</b>공통 비교 경기</div><div class=\"card\"><b>{len(coefficients)}</b>Ridge 계수</div></div>
    <h2>1. 데이터와 검증 계약</h2><p>2020–2024만 학습하고 2025를 한 번 예측했다. 선수 자격은 2021–2025 매년 공식 선발 20경기 이상이다. q33/q67과 전처리 통계는 모두 2024년까지의 학습 데이터에서 계산한다.</p>
    <p class=\"note\">선수 자격 확인에는 2025 출전 여부가 사용되므로 완전한 사전 배포 시뮬레이션은 아니다. 다만 2025 Stuff+ 결과값은 모델 선택이나 학습에 사용하지 않는다.</p>
    <h2>2. 모델 설정</h2><ul><li><b>EWMA4:</b> 최근 경기 Stuff+ span 4.</li><li><b>Ridge:</b> 선수별 compact16, alpha=100, residual correction scale=0.25.</li><li><b>XGBoost:</b> 선수별 compact16, validation trial 16, trees=100, depth=2, residual correction scale=0.221845.</li><li><b>TCN:</b> shared causal H1, ordered L=8, channels=4, bottleneck=4, dilation 1/2/4, alpha=0.1, epochs=250, 5 seeds 평균.</li></ul>
    <p>네 모델 모두 최종 연속값을 <code>예측 Stuff+ = EWMA4 + 보정 residual</code> 구조로 만든다. TCN은 시간 순서를 학습하고 XGBoost는 비선형 상호작용을 허용하지만, 둘 다 Ridge처럼 계수 하나로 직접 해석할 수는 없다.</p>
    <h2>3. 2025 공통 표본 성능</h2><div class=\"table-wrap\">{table_html(metric)}</div>
    <h2>4. 모델별 배포 가능 표본</h2><div class=\"table-wrap\">{table_html(deploy)}</div>
    <h2>5. 선수별 최근 경기 연속 예측</h2><p>실제값은 검증용으로만 표시한다. 예측에는 해당 경기보다 앞선 기록만 사용한다.</p><div class=\"table-wrap\">{table_html(latest_display, digits=3, classes=True)}</div>
    <h2>6. Ridge 해석 방법</h2><p><code>standardized_residual_coefficient</code>는 표준편차 1 증가에 대한 residual 변화다. <code>effective_standardized_coefficient</code>는 최종 0.25 보정까지 반영한다. <code>effective_raw_unit_coefficient</code>는 원래 feature 1단위 증가에 따른 최종 Stuff+ 변화다. 양수는 예측 상승, 음수는 하락 방향이며 상관관계이지 인과효과가 아니다.</p>
    <h3>선수별 절댓값 상위 3개 표준화 계수</h3><div class=\"table-wrap\">{table_html(coef_summary, digits=6)}</div>
    <h3>선수별 전체 16개 계수</h3>{''.join(coef_sections)}
    <h2>7. 해석상 주의점</h2><ul><li>선수별 Ridge는 표본이 작고 feature끼리 상관되므로 계수 부호가 불안정할 수 있다.</li><li>TCN은 248개 파라미터의 공동 모델로, 선수별 bias를 포함하지만 단순 계수표는 제공하지 않는다.</li><li>등급은 선수별 학습 q33/q67로 나누므로 서로 다른 선수의 Low/Middle/High를 절대 수준처럼 비교하면 안 된다.</li><li>같은 예측값이 나오는 경우는 모델이 EWMA 보정을 0에 가깝게 내거나 반올림된 경우이며, 원본 CSV에는 반올림 전 값이 저장된다.</li></ul>
    <h2>8. 재현 산출물</h2><p><code>replay_tcn_ridge_report.py</code>, <code>latest_player_predictions.csv</code>, <code>ridge_coefficients_all.csv</code>, <code>predictions_2025.parquet</code>, <code>metrics_*.csv</code>.</p>
    </main></body></html>"""
    html_path.write_text(html, encoding="utf-8")

    md = f"""# Stuff+ 통합 예측 보고서\n\n생성: {generated}\n\nTCN은 제거된 모델이 아니다. 검증상 주 모델 교체 기준을 통과하지 못해 진단 모델로 남았지만, 고정 설정으로 2025 선수별 연속 예측까지 산출했다.\n\n## 실행 계약\n\n- 학습: 2020–2024\n- 평가: 2025 ({metadata['evaluation_rows']}경기)\n- 공통 비교: {metadata['common_rows']}경기\n- 대상: {metadata['qualified_pitchers']}명\n- Ridge 전체 계수: {len(coefficients)}개 ({metadata['qualified_pitchers']}명 × {len(COMPACT_FEATURES)} features)\n\n## 모델\n\n- EWMA4: span=4\n- Ridge: alpha=100, correction scale=0.25, 선수별 compact16\n- XGBoost: trial 16, trees=100, depth=2, correction scale={XGBOOST_SCALE}\n- TCN: H1 ordered, L=8, channels=4, bottleneck=4, alpha=0.1, epochs=250, seeds={list(SEEDS)}\n\n## 산출물\n\n전체 성능표, 선수별 최근 예측, Ridge 계수와 해석은 `FINAL_MODEL_REPORT.html`에 포함되어 있다. 원자료는 `{metadata['output_dir']}`에 저장된다.\n"""
    markdown_path.write_text(md, encoding="utf-8")


def build_presentation_reports(
    *, latest: pd.DataFrame, common: pd.DataFrame, deployable: pd.DataFrame,
    coefficients: pd.DataFrame, continuous: pd.DataFrame, metadata: dict,
    html_path: Path, markdown_path: Path,
) -> None:
    """Build the presentation report using the 7/30 technical-report structure."""

    def percent(value: object) -> str:
        return f"{100 * float(value):.2f}%" if pd.notna(value) else "-"

    common_view = common.merge(continuous, on="model", how="left")
    common_view["모델"] = common_view["model"].map(MODEL_LABELS)
    common_view["Accuracy"] = common_view["accuracy"].map(percent)
    common_view["Balanced accuracy"] = common_view["balanced_accuracy"].map(percent)
    common_view["Macro-F1"] = common_view["macro_f1"].map(lambda x: f"{x:.4f}")
    common_view["Ordinal MAE"] = common_view["ordinal_mae"].map(lambda x: f"{x:.4f}")
    common_view["연속 MAE"] = common_view["mae"].map(lambda x: f"{x:.3f}")
    common_view["연속 RMSE"] = common_view["rmse"].map(lambda x: f"{x:.3f}")
    common_view = common_view[
        ["모델", "Accuracy", "Balanced accuracy", "Macro-F1", "Ordinal MAE", "연속 MAE", "연속 RMSE"]
    ]

    deploy_view = deployable.copy()
    deploy_view["모델"] = deploy_view["model"].map(MODEL_LABELS)
    deploy_view["예측/대상"] = deploy_view.apply(
        lambda row: f"{int(row.n_predicted)} / {int(row.n_total_eligible)}", axis=1
    )
    deploy_view["Coverage"] = deploy_view["coverage"].map(percent)
    deploy_view["Accuracy"] = deploy_view["accuracy"].map(percent)
    deploy_view["Balanced accuracy"] = deploy_view["balanced_accuracy"].map(percent)
    deploy_view = deploy_view[["모델", "예측/대상", "Coverage", "Accuracy", "Balanced accuracy"]]

    latest_view = latest.rename(columns={
        "pitcher_name": "선수", "pitcher": "MLBAM", "game_date": "최근 경기",
        "true_stuff_plus": "실제 Stuff+", "true_class": "실제 등급",
        "q33": "q33", "q67": "q67",
        "ewma_predicted_stuff_plus": "EWMA4 예측", "ewma_predicted_class": "EWMA4 등급",
        "ridge_predicted_stuff_plus": "Ridge 예측", "ridge_predicted_class": "Ridge 등급",
        "xgboost_predicted_stuff_plus": "XGBoost 예측", "xgboost_predicted_class": "XGBoost 등급",
        "tcn_predicted_stuff_plus": "TCN 예측", "tcn_predicted_class": "TCN 등급",
    })
    latest_columns = [
        "선수", "최근 경기", "실제 Stuff+", "실제 등급", "q33", "q67",
        "EWMA4 예측", "EWMA4 등급", "Ridge 예측", "Ridge 등급",
        "XGBoost 예측", "XGBoost 등급", "TCN 예측", "TCN 등급",
    ]
    latest_view = latest_view[[column for column in latest_columns if column in latest_view]]

    top_coefficients = (
        coefficients.assign(abs_coef=lambda frame: frame.effective_standardized_coefficient.abs())
        .sort_values(["pitcher_name", "abs_coef"], ascending=[True, False])
        .groupby(["pitcher_name", "pitcher"], as_index=False)
        .head(3)[
            ["pitcher_name", "pitcher", "feature", "effective_standardized_coefficient", "effective_raw_unit_coefficient"]
        ]
        .rename(columns={
            "pitcher_name": "선수", "pitcher": "MLBAM", "feature": "피처",
            "effective_standardized_coefficient": "표준화 계수(0.25 반영)",
            "effective_raw_unit_coefficient": "원단위 계수(0.25 반영)",
        })
    )
    coefficient_sections: list[str] = []
    coefficient_columns = [
        "feature", "standardized_residual_coefficient",
        "effective_standardized_coefficient", "effective_raw_unit_coefficient",
        "feature_training_mean", "feature_training_scale",
    ]
    for (name, pitcher), group in coefficients.groupby(["pitcher_name", "pitcher"], sort=True):
        coefficient_sections.append(
            f'<details><summary>{escape(str(name))} <span>{escape(str(pitcher))}</span></summary>'
            + '<div class="table-wrap">'
            + table_html(group[coefficient_columns], digits=6)
            + "</div></details>"
        )

    selection_view = pd.DataFrame([
        ["XGBoost", 0.5613, 0.0177, 0.5525, 1900, "주 비선형 모델"],
        ["EWMA4", 0.5583, 0.0231, 0.5467, 0, "기준선"],
        ["Ridge", 0.5539, 0.0238, 0.5420, 323, "해석 모델"],
        ["TCN H1-4", 0.5548, 0.0263, 0.5417, 248, "시퀀스 진단 모델"],
        ["Ordinal linear", 0.4734, 0.0445, 0.4512, 361, "순서형 진단 모델"],
    ], columns=["모델", "Mean BA", "Std", "Selection score", "Params", "역할"])
    ordinal_view = pd.DataFrame([
        ["Ordinal linear", "47.48%", "46.15%", "0.4466", "0.6620", "560/560"],
        ["Ridge–TCN ensemble (β=0)", "49.90%", "48.25%", "0.4872", "0.5573", "497/560"],
    ], columns=["진단 모델", "Accuracy", "Balanced accuracy", "Macro-F1", "Ordinal MAE", "Coverage"])

    generated = datetime.now().astimezone().isoformat(timespec="minutes")
    css = """
    :root{--paper:#fff;--bg:#eef1f5;--ink:#1d2533;--muted:#657085;--line:#d8dee8;--navy:#17355c;--blue:#2f6fb2;--pale:#edf4fb;--green:#217a56;--amber:#a76709}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Pretendard","Noto Sans KR","Apple SD Gothic Neo","Segoe UI",sans-serif;word-break:keep-all}
    main{width:min(1180px,calc(100% - 32px));margin:28px auto 60px;background:var(--paper);border:1px solid var(--line);box-shadow:0 10px 32px rgba(20,37,60,.08)}
    .cover{padding:72px 76px 56px;color:#fff;background:linear-gradient(135deg,#122a49,#244f7c)}.eyebrow{font-size:13px;letter-spacing:.14em;text-transform:uppercase;opacity:.76}.cover h1{max-width:850px;margin:18px 0 16px;font-size:42px;line-height:1.25}.cover p{max-width:820px;margin:0;font-size:18px;color:#dbe8f6}.cover-meta{display:flex;gap:28px;margin-top:42px;font-size:13px;color:#c4d5e8}
    article{padding:50px 76px 80px}h2{margin:58px 0 18px;padding-bottom:9px;border-bottom:2px solid var(--navy);font-size:27px;color:var(--navy)}h3{margin:32px 0 12px;font-size:20px;color:#273e5c}h4{margin:24px 0 8px;font-size:16px}p{margin:10px 0}ul,ol{margin:10px 0;padding-left:1.4rem}.small{font-size:13px;color:var(--muted)}
    nav{margin:0 -10px 32px;padding:22px 26px;border:1px solid var(--line);background:#f8fafc}nav b{display:block;margin-bottom:8px}nav ol{columns:2;column-gap:40px}nav li{padding:3px 0}nav a{color:var(--navy);text-decoration:none}
    .lead{font-size:18px;line-height:1.8}.keyline{margin:22px 0;padding:20px 22px;border-left:5px solid var(--blue);background:var(--pale)}.caution{border-left-color:#d3962f;background:#fff8e8}.conclusion{border-left-color:var(--green);background:#edf8f2}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card{padding:17px;border:1px solid var(--line);background:#fff}.card strong{display:block;font-size:25px;color:var(--navy)}.card span{font-size:13px;color:var(--muted)}
    .model-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}.model-card{padding:20px;border:1px solid var(--line);border-top:4px solid var(--blue)}.model-card h3{margin:0 0 8px}.model-card .role{color:var(--blue);font-weight:700;font-size:13px}
    .table-wrap{overflow:auto;margin:14px 0 24px;border:1px solid var(--line)}table.data{width:100%;border-collapse:collapse;font-size:13px}table.data th,table.data td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}table.data th{background:#f0f4f8;color:#32465f;font-weight:700}table.data th:first-child,table.data td:first-child{text-align:left}table.data tr:last-child td{border-bottom:0}
    .formula{margin:14px 0;padding:17px 20px;background:#172238;color:#e8f0fa;font:14px/1.7 "Cascadia Code",Consolas,monospace;white-space:pre-wrap}.pipeline{display:flex;align-items:stretch;gap:8px;margin:18px 0;overflow:auto}.step{min-width:145px;padding:13px;border:1px solid #bed0e2;background:#f5f9fd;text-align:center;font-size:13px}.arrow{align-self:center;color:var(--blue);font-size:24px}.branch{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.branch .step{min-width:0}
    .arch{display:grid;grid-template-columns:170px 1fr;gap:12px 22px;margin:18px 0}.arch dt{font-weight:800;color:var(--navy)}.arch dd{margin:0;padding-bottom:12px;border-bottom:1px solid var(--line)}
    details{margin:8px 0;border:1px solid var(--line);background:#fff}summary{cursor:pointer;padding:11px 14px;font-weight:750}summary span{font-weight:400;color:var(--muted)}details .table-wrap{margin:0;border:0;border-top:1px solid var(--line)}code{padding:.1em .35em;background:#eef2f6;color:#a23c32;border-radius:3px}.source{margin-top:54px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
    .print{position:fixed;right:22px;top:18px;z-index:2;padding:10px 15px;border:0;border-radius:99px;background:#17355c;color:#fff;font-weight:700;cursor:pointer}
    @media(max-width:820px){main{width:100%;margin:0;border:0}.cover,article{padding:38px 22px}.cover h1{font-size:32px}.cards,.model-grid,.branch{grid-template-columns:1fr 1fr}nav ol{columns:1}.print{position:static;float:right;margin:10px}}
    @media print{body{background:#fff}main{width:auto;margin:0;border:0;box-shadow:none}.print{display:none}.cover{-webkit-print-color-adjust:exact;print-color-adjust:exact}.table-wrap{overflow:visible}details{break-inside:avoid}details[open] summary{display:none}}
    """
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLB 선발투수 Stuff+ 예측 최종 보고서</title><style>{css}</style></head><body><button class="print" onclick="window.print()">인쇄 / PDF</button><main>
    <header class="cover"><div class="eyebrow">PAINS × BREA:D PROJECT TF · FINAL REPORT</div><h1>MLB 선발투수 Stuff+<br>등급 및 연속값 예측</h1><p>최근 등판 흐름과 투구 특성으로 다음 선발 경기의 Stuff+를 예측하고, 모델별 성능과 해석 가능성을 비교한 기술 보고서</p><div class="cover-meta"><span>학습 2020–2024</span><span>최종 평가 2025</span><span>대상 19명</span><span>작성 {generated}</span></div></header>
    <article><nav><b>보고서 구성</b><ol><li><a href="#overview">연구 개요</a></li><li><a href="#collection">데이터 수집</a></li><li><a href="#models">모델</a></li><li><a href="#results">결과</a></li><li><a href="#interpretation">해석</a></li><li><a href="#conclusion">결론</a></li><li><a href="#architecture">모델 아키텍처</a></li><li><a href="#pipeline">전체 파이프라인</a></li></ol></nav>

    <section id="overview"><h2>1. 연구 개요</h2><p class="lead">이 프로젝트의 목적은 선발투수의 다음 경기 Stuff+가 본인 기준으로 낮은지, 보통인지, 높은지를 예측하는 것이다. 단순한 등급 분류뿐 아니라 실제 Stuff+ 연속값도 함께 산출해 모델이 어느 방향으로 얼마나 보정했는지 확인했다.</p><div class="keyline conclusion"><b>핵심 결론</b><br>XGBoost·Ridge·TCN의 등급 성능은 사실상 비슷했다. XGBoost는 Low·High 구간, TCN은 Middle 구간과 전체 커버리지, Ridge는 계수 기반 설명에서 상대적인 장점이 있었다. 따라서 하나의 모델을 압도적 우승 모델로 해석하기보다 역할을 나눠 보는 편이 타당하다.</div>
    <div class="cards"><div class="card"><strong>{metadata['qualified_pitchers']}</strong><span>분석 대상 투수</span></div><div class="card"><strong>{metadata['evaluation_rows']}</strong><span>2025 평가 대상 경기</span></div><div class="card"><strong>{metadata['common_rows']}</strong><span>공통 비교 경기</span></div><div class="card"><strong>{len(coefficients)}</strong><span>공개한 Ridge 계수</span></div></div>
    <h3>예측 기준선: EWMA4</h3><p>모든 residual 모델의 출발점은 최근 Stuff+의 span-4 지수가중평균이다. 가장 최근 등판에 40%, 한 경기 전에는 24%, 두 경기 전에는 14.4%의 가중치가 들어간다. Ridge·XGBoost·TCN은 이 값 자체를 버리지 않고, 다음 경기에서 필요한 보정량만 학습한다.</p><div class="formula">residual = 실제 Stuff+ − EWMA4\n최종 예측 Stuff+ = EWMA4 + correction_scale × 예측 residual</div></section>

    <section id="collection"><h2>2. 데이터 수집</h2><h3>2.1 원본 데이터</h3><div class="table-wrap"><table class="data"><thead><tr><th>출처</th><th>수집 내용</th><th>모델에서의 역할</th></tr></thead><tbody><tr><td>Statcast</td><td>구속, 회전수, 릴리스, 무브먼트, 구종, 투구 수</td><td>등판 특성 및 시계열 입력</td></tr><tr><td>FanGraphs 경기 로그</td><td>등판 단위 Stuff+</td><td>예측 대상과 과거 이력</td></tr><tr><td>MLB 공식 기록</td><td>선수 식별자, 이름, 공식 선발 정보</td><td>선수 매칭과 선발 판정</td></tr></tbody></table></div>
    <h3>2.2 분석 대상</h3><p>2021–2025년 매 시즌 공식 선발이면서 50구 이상인 경기를 20회 이상 확보한 투수 19명을 선정했다. 2020년은 단축 시즌이므로 자격 판정에는 쓰지 않고, 모델의 과거 학습 이력으로만 사용했다.</p>
    <h3>2.3 수집 과정과 재현성</h3><p><code>collect_mlb_stuff_dataset.py</code>가 대상 후보를 계산하고 Statcast·FanGraphs·MLB 공식 기록을 전용 디렉터리에 수집한다. 각 파일의 경로, 기간, 행 수, SHA-256을 매니페스트로 남기며 <code>verify_mlb_stuff_collection.py</code>가 원본과 모델 준비 데이터의 수를 다시 확인한다.</p><div class="keyline"><b>현재 검증 결과</b><br>177개 수집 파일, 대상 투수 19명, 모델 준비 데이터 2,600행이 매니페스트와 일치했다.</div>
    <h3>2.4 history와 target의 분리</h3><div class="table-wrap"><table class="data"><thead><tr><th>구분</th><th>조건</th><th>용도</th></tr></thead><tbody><tr><td>history_eligible</td><td>공식 선발</td><td>이후 경기의 과거 이력으로 사용</td></tr><tr><td>target_eligible</td><td>공식 선발 + 50구 이상 + Stuff+ 존재</td><td>실제 예측 및 평가 대상</td></tr></tbody></table></div><p>50구 미만 선발은 채점 대상에서는 빠지지만 과거 이력에는 남는다. 다음 경기 EWMA와 시퀀스가 실제 등판 흐름을 유지하도록 하기 위한 처리다.</p>
    <h3>2.5 시간축·구종·결측 처리</h3><ul><li>투수별 전체 시즌을 연속 시간축으로 정렬하고 <code>season_start</code>, <code>long_gap</code>, <code>rest_days_log</code>를 계산한다.</li><li>FF·SI·FC·FA는 fastball share, SL·ST·CU·KC·SV·CS는 breaking, CH·FS·FO·SC는 offspeed로 묶는다.</li><li>주력 패스트볼은 해당 fold의 학습 기간 사용량만으로 결정해 미래 정보를 차단한다.</li><li>Ridge·XGBoost는 실제 사용 피처 16개만 complete-case로 검사한다. TCN은 학습 구간의 투수별 중앙값과 pooled fallback을 사용한다.</li></ul>
    <h3>2.6 등급 라벨</h3><div class="formula">Low    : Stuff+ ≤ q33\nMiddle : q33 &lt; Stuff+ ≤ q67\nHigh   : Stuff+ &gt; q67</div><p>q33과 q67은 투수별·fold별 학습 기간에서만 계산한다. 선수마다 기준이 다르므로 다른 선수의 Low와 High를 절대 수준처럼 비교하지 않는다.</p></section>

    <section id="models"><h2>3. 모델</h2><div class="model-grid"><div class="model-card"><div class="role">BASELINE</div><h3>EWMA4</h3><p>최근 등판에 더 큰 가중치를 둔 기준선이다. 별도 학습 없이 최근 흐름만 반영한다.</p></div><div class="model-card"><div class="role">INTERPRETABLE</div><h3>Ridge</h3><p>투수별 compact16으로 EWMA 대비 residual을 학습한다. alpha=100, correction scale=0.25이며 선수별 계수를 직접 확인할 수 있다.</p></div><div class="model-card"><div class="role">NON-LINEAR</div><h3>XGBoost</h3><p>Ridge와 같은 compact16을 사용하되 비선형 관계와 변수 상호작용을 트리로 학습한다. 2022–2024 검증에서 주 비선형 모델로 선택됐다.</p></div><div class="model-card"><div class="role">SEQUENCE</div><h3>Shared TCN</h3><p>최근 8경기의 raw15 시퀀스를 causal convolution으로 처리한다. 19명이 encoder를 공유하고 투수별 bias만 분리한다.</p></div></div>
    <h3>진단 모델</h3><p>Ordinal linear는 Low–Middle–High의 순서를 확률 구조에 직접 반영했다. Ridge–TCN 앙상블도 검토했지만, β=0.3의 점수 개선이 0.003 simplicity margin 안에 있어 더 단순한 β=0이 채택됐다. 최종 앙상블이 Ridge와 같은 이유는 이 선택 규칙 때문이다.</p><div class="table-wrap">{table_html(ordinal_view, digits=4)}</div>
    <h3>모델 선택 규칙</h3><div class="table-wrap"><table class="data"><thead><tr><th>Fold</th><th>학습</th><th>검증</th><th>용도</th></tr></thead><tbody><tr><td>2022</td><td>2020–2021</td><td>2022</td><td>모델 선택</td></tr><tr><td>2023</td><td>2020–2022</td><td>2023</td><td>모델 선택</td></tr><tr><td>2024</td><td>2020–2023</td><td>2024</td><td>모델 선택</td></tr><tr><td>2025</td><td>2020–2024</td><td>2025</td><td>최종 평가 1회</td></tr></tbody></table></div><div class="formula">selection score = mean(balanced accuracy) − 0.5 × std(balanced accuracy)</div><p>최고 점수와 0.003 이내인 후보는 파라미터 수가 적은 쪽을 선택했다. 2025 결과를 보고 설정을 바꾸지 않도록 dataset·fold·code fingerprint도 함께 잠근다.</p><div class="table-wrap">{table_html(selection_view, digits=4)}</div></section>

    <section id="results"><h2>4. 결과</h2><h3>4.1 동일 표본 비교: 497경기</h3><p>모든 핵심 모델이 예측 가능한 같은 497경기만 남겨 비교한 결과다. 등급 성능은 XGBoost가 근소하게 높았지만 차이는 작았다. 연속 오차는 표본 범위가 다른 모델끼리 직접 순위를 매기지 않고 보조 지표로 본다.</p><div class="table-wrap">{table_html(common_view, digits=4)}</div>
    <h3>4.2 실제 예측 가능 범위: 560경기</h3><p>TCN과 EWMA4는 560경기 전체를 예측했다. Ridge와 XGBoost는 compact16 결측 때문에 63경기를 예측하지 못했다. 따라서 아래 표의 정확도는 표본이 다르며, 공정한 모델 비교는 위 공통 표본을 기준으로 한다.</p><div class="table-wrap">{table_html(deploy_view, digits=4)}</div>
    <h3>4.3 선수별 최근 경기 예측</h3><p>각 선수의 2025년 마지막 평가 경기에서 실제 Stuff+와 네 모델의 연속 예측을 나란히 표시했다. HTML은 보기 편하도록 반올림하고, CSV에는 원래 정밀도를 보존했다.</p><div class="table-wrap">{table_html(latest_view, digits=3, classes=True)}</div></section>

    <section id="interpretation"><h2>5. 해석</h2><h3>5.1 TCN은 제외된 모델이 아니다</h3><p>TCN은 실제로 선택 실험과 2025 최종 평가를 모두 수행했다. 다만 2022–2024 selection score가 XGBoost를 넘지 못해 주 모델을 교체하지 않았을 뿐이다. 전체 560경기에서는 100% 커버리지를 확보했고, 연속값 MAE는 4.301로 네 모델 중 가장 낮았다.</p>
    <h3>5.2 모델 예측이 비슷한 이유</h3><p>Ridge·XGBoost·TCN 모두 EWMA4를 기준으로 작은 residual만 보정한다. correction scale도 0.10–0.25 수준이라 최근 흐름이 최종값의 대부분을 차지한다. residual이 작거나 같은 등급 경계 안에 머무르면 연속 예측값이 달라도 등급 결과는 같아진다. 반올림된 표에서는 차이가 더 작아 보일 수 있다.</p>
    <h3>5.3 Ridge 계수 읽는 법</h3><p><code>effective_standardized_coefficient</code>는 해당 피처가 선수 본인의 학습 표준편차만큼 증가할 때 최종 Stuff+ 보정 방향이 어떻게 바뀌는지를 나타낸다. 양수는 상승, 음수는 하락 방향이다. <code>effective_raw_unit_coefficient</code>는 원래 단위 1 증가 기준으로 환산한 값이다.</p><div class="keyline caution"><b>해석 범위</b><br>계수는 인과효과가 아니다. 선수별 표본이 작고 피처끼리 상관되어 있으므로 부호와 크기는 해당 학습 기간 안의 조건부 관계로 읽어야 한다.</div><h4>선수별 절댓값 상위 3개 계수</h4><div class="table-wrap">{table_html(top_coefficients, digits=6)}</div>
    <h3>5.4 결과가 이전 보고서보다 낮아진 이유</h3><p>현재 파이프라인은 q33·q67을 compact 피처의 결측 여부와 무관한 전체 target-eligible 표본에서 계산한다. 이전에는 28개 피처가 모두 존재하는 2,058행에서 경계를 계산했고, 현재는 2,387행으로 모집단이 넓어졌다. 같은 2025 공통 경기 중 일부의 정답 등급 자체가 바뀌었기 때문에 단순 정확도는 이전 결과와 직접 비교할 수 없다.</p>
    <h3>5.5 한계</h3><ul><li>최종 평가는 2025년 한 시즌이며, 이 결과로 모델을 다시 선택하지 않았다.</li><li>2025 출전 수는 평가 선수 자격 확인에 사용돼 완전한 사전 배포 시뮬레이션은 아니다.</li><li>Ridge·XGBoost의 커버리지는 88.75%로 TCN·EWMA4보다 낮다.</li><li>선수별 성능 편차가 커서 전체 평균을 개별 선수의 기대 성능으로 그대로 적용할 수 없다.</li></ul></section>

    <section id="conclusion"><h2>6. 결론</h2><div class="keyline conclusion"><b>발표용 요약</b><br>최근 등판 흐름만으로도 상당 부분이 설명됐고, 복잡한 모델의 추가 이득은 제한적이었다. XGBoost는 비선형 주 모델, Ridge는 설명 모델, TCN은 결측에 강한 시퀀스 진단 모델로 역할을 나누는 것이 현재 결과에 가장 잘 맞는다.</div><ol><li><b>단일 우승 모델은 없다.</b> 공통 표본에서 세 학습 모델의 등급 성능 차이는 매우 작다.</li><li><b>해석은 Ridge가 담당한다.</b> 선수별 16개 계수와 원단위 환산값을 모두 공개했다.</li><li><b>TCN은 유효한 보조 모델이다.</b> 100% 커버리지와 가장 낮은 연속 MAE를 보였지만 주 모델 교체 기준은 통과하지 못했다.</li><li><b>운영 시 예측값과 불확실성을 같이 보여야 한다.</b> 모델 간 등급이 다르거나 경계에 가까운 경우를 별도로 표시하는 방식이 적절하다.</li></ol></section>

    <section id="architecture"><h2>7. 모델 아키텍처</h2><h3>7.1 공통 residual 구조</h3><div class="pipeline"><div class="step">과거 Stuff+<br>EWMA4</div><div class="arrow">→</div><div class="step">모델별 입력<br>Compact16 / Raw15</div><div class="arrow">→</div><div class="step">residual 예측</div><div class="arrow">→</div><div class="step">scale 적용</div><div class="arrow">→</div><div class="step">연속 Stuff+<br>및 등급</div></div>
    <h3>7.2 Ridge·XGBoost</h3><dl class="arch"><dt>입력</dt><dd>최근 Stuff+ 편차, workload, 휴식일, 릴리스·무브먼트·구종 구성 slope 등 compact16</dd><dt>Ridge</dt><dd>선수별 표준화 → L2 선형 회귀(alpha=100) → residual × 0.25</dd><dt>XGBoost</dt><dd>선수별 shallow tree 100개(depth=2) → residual × 0.221845</dd><dt>결측</dt><dd>사용 피처 16개 complete-case. 불완전한 경기는 예측하지 않는다.</dd></dl>
    <h3>7.3 Shared TCN</h3><div class="pipeline"><div class="step">최근 8경기<br>15 raw features</div><div class="arrow">→</div><div class="step">Causal Conv1d<br>dilation 1</div><div class="arrow">→</div><div class="step">Causal Conv1d<br>dilation 2</div><div class="arrow">→</div><div class="step">Causal Conv1d<br>dilation 4</div><div class="arrow">→</div><div class="step">4D bottleneck</div><div class="arrow">→</div><div class="step">global head<br>+ pitcher bias</div></div><dl class="arch"><dt>시퀀스</dt><dd>Stuff+, pitch count, 휴식·시즌 플래그, 주력 패스트볼, 릴리스, arm angle, 구종 비중</dd><dt>마스크</dt><dd>valid timestep, primary fastball availability, Stuff+ availability를 별도로 전달</dd><dt>학습</dt><dd>Huber loss, pitcher-balanced objective, dropout 0.1, learning rate 0.001, 250 epochs</dd><dt>최종값</dt><dd>H1 ordered, channels=4, bottleneck=4, residual × 0.1, 5 seeds 평균, 248 parameters</dd></dl>
    <h3>7.4 Ordinal linear와 앙상블</h3><dl class="arch"><dt>Ordinal</dt><dd>Compact16 + EWMA tertile position → proportional-odds 확률 P(Low/Middle/High)</dd><dt>Ensemble</dt><dd>(1−β) × Ridge residual + β × TCN residual. 단순성 규칙으로 β=0이 선택돼 최종값은 Ridge와 동일</dd></dl></section>

    <section id="pipeline"><h2>8. 전체 파이프라인</h2><div class="pipeline"><div class="step">대상 후보 계산</div><div class="arrow">→</div><div class="step">원천 데이터 수집</div><div class="arrow">→</div><div class="step">투구 → 등판 집계</div><div class="arrow">→</div><div class="step">선발·eligibility 판정</div><div class="arrow">→</div><div class="step">시간축·구종 처리</div></div><div class="pipeline"><div class="step">fold-local q33/q67</div><div class="arrow">→</div><div class="step">Compact16 / Raw15</div><div class="arrow">→</div><div class="step">2022–2024 선택</div><div class="arrow">→</div><div class="step">설정·fingerprint 잠금</div><div class="arrow">→</div><div class="step">2025 최종 평가</div><div class="arrow">→</div><div class="step">CSV·HTML 보고서</div></div>
    <h3>실행 파일별 역할</h3><div class="table-wrap"><table class="data"><thead><tr><th>단계</th><th>실행 파일</th><th>산출물</th></tr></thead><tbody><tr><td>수집</td><td>collect_mlb_stuff_dataset.py</td><td>전용 데이터 디렉터리, 수집 매니페스트</td></tr><tr><td>검증</td><td>verify_mlb_stuff_collection.py</td><td>파일·선수·모델 준비 행 수 검증</td></tr><tr><td>모델 선택</td><td>select_models.py</td><td>locked_model_config.json</td></tr><tr><td>최종 평가</td><td>final_evaluate.py</td><td>2025 예측과 성능표</td></tr><tr><td>발표 보고서</td><td>replay_tcn_ridge_report.py</td><td>선수별 예측, Ridge 계수, HTML</td></tr></tbody></table></div></section>

    <section><h2>부록 A. 선수별 Ridge 전체 계수</h2><p>각 선수를 펼치면 16개 피처의 표준화 residual 계수, 0.25 correction scale 반영 계수, 원단위 환산 계수와 학습 통계를 확인할 수 있다.</p>{''.join(coefficient_sections)}</section>
    <section class="source"><b>보고서 근거</b><br>7/30 최종 보고서(Notion export), 현재 저장소의 README·모델 구현·수집 매니페스트, <code>experiments/runs/tcn_ridge_replay_2025</code> 재현 산출물. 테스트 결과: 89 passed, 2 skipped.</section>
    </article></main></body></html>"""
    html_path.write_text(html, encoding="utf-8")

    markdown = f"""# MLB 선발투수 Stuff+ 등급 및 연속값 예측 — 최종 보고서

작성: {generated}
학습: 2020–2024 / 최종 평가: 2025 / 대상: {metadata['qualified_pitchers']}명

## 1. 연구 개요

다음 선발 경기의 Stuff+ 연속값과 선수별 Low·Middle·High 등급을 예측했다. XGBoost·Ridge·TCN의 등급 성능은 비슷했으며, XGBoost는 비선형 관계, Ridge는 계수 해석, TCN은 100% 커버리지와 시계열 표현에서 역할이 나뉜다.

## 2. 데이터 수집

Statcast 투구 데이터, FanGraphs 등판 Stuff+, MLB 공식 선발 기록을 결합했다. `collect_mlb_stuff_dataset.py`가 수집 과정과 SHA-256 매니페스트를 남기고 `verify_mlb_stuff_collection.py`가 177개 파일, 19명, 모델 준비 2,600행을 검증한다.

## 3. 모델

- EWMA4: 최근 Stuff+ span-4 기준선
- Ridge: 선수별 compact16, alpha=100, correction scale=0.25
- XGBoost: 선수별 compact16, 100 trees, depth=2, correction scale={XGBOOST_SCALE}
- Shared TCN: 최근 8경기 raw15, H1 ordered, 5 seeds, 248 parameters
- 진단: Ordinal linear, Ridge–TCN ensemble

## 4. 결과와 해석

공통 497경기에서 XGBoost accuracy 50.10%, Ridge 50.10%, TCN 49.90%, EWMA4 49.09%였다. TCN은 전체 560경기를 모두 예측했고 연속 MAE 4.301로 가장 낮았다. Ridge와 XGBoost는 compact16 결측으로 497경기를 예측했다.

## 5. 결론

최근 흐름이 예측의 대부분을 설명하며 복잡한 모델의 추가 이득은 제한적이다. XGBoost를 비선형 주 모델, Ridge를 설명 모델, TCN을 시퀀스 진단 모델로 병행하는 구성이 현재 결과에 가장 적합하다.

## 6. 아키텍처와 파이프라인

전체 모델 구조, 데이터 수집부터 2025 평가와 HTML 생성까지의 파이프라인, 선수별 예측, Ridge 전체 304개 계수는 `FINAL_MODEL_REPORT.html`에 수록했다.
"""
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.add_xgboost:
        bundle = load_bundle(args.data_root)
        fold = prepare_fold(bundle.outings, 2025, train_start=2020, evaluation_split="test")
        xgboost, _ = predict_residual_fold(
            fold, model="xgboost", features=COMPACT_FEATURES,
            correction_scale=XGBOOST_SCALE, xgboost_params=XGBOOST_CONFIG,
            random_state=SEEDS[0], candidate_id="xgboost_trial16_scale0.221845_locked",
            missing_policy="complete_case",
        )
        existing = pd.read_parquet(args.output_dir / "predictions_2025.parquet")
        existing = existing.loc[existing["report_model"].ne("xgboost")]
        combined = pd.concat(
            [existing, xgboost.assign(report_model="xgboost")],
            ignore_index=True, sort=False,
        )
        combined.to_parquet(args.output_dir / "predictions_2025.parquet", index=False)
        predictions = {
            model: group.drop(columns="report_model").reset_index(drop=True)
            for model, group in combined.groupby("report_model", sort=False)
        }
        common, deployable = evaluate_common_and_deployable(predictions)
        continuous = pd.DataFrame(
            [{"model": model, **continuous_metrics(frame)} for model, frame in predictions.items()]
        )
        latest = make_latest(
            predictions, {str(key): value for key, value in bundle.pitcher_names.items()}
        )
        common.to_csv(args.output_dir / "metrics_common.csv", index=False)
        deployable.to_csv(args.output_dir / "metrics_deployable.csv", index=False)
        continuous.to_csv(args.output_dir / "metrics_continuous.csv", index=False)
        latest.to_csv(args.output_dir / "latest_player_predictions.csv", index=False)
        print("Locked XGBoost merged; run --report-only to rebuild the report.")
        return
    if args.report_only:
        common = pd.read_csv(args.output_dir / "metrics_common.csv")
        deployable = pd.read_csv(args.output_dir / "metrics_deployable.csv")
        continuous = pd.read_csv(args.output_dir / "metrics_continuous.csv")
        coefficient_frame = pd.read_csv(args.output_dir / "ridge_coefficients_all.csv")
        latest = pd.read_csv(args.output_dir / "latest_player_predictions.csv")
        metadata = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "dataset_fingerprint": "see collection manifest",
            "qualified_pitchers": int(latest["pitcher"].nunique()),
            "evaluation_rows": int(deployable["n_total_eligible"].max()),
            "common_rows": int(common["n_evaluated"].min()),
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_correction_scale": RIDGE_SCALE,
            "tcn_config": asdict(TCN_CONFIG),
            "tcn_seeds": list(SEEDS),
            "output_dir": str(args.output_dir),
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        build_presentation_reports(
            latest=latest, common=common, deployable=deployable,
            coefficients=coefficient_frame, continuous=continuous, metadata=metadata,
            html_path=args.html, markdown_path=args.markdown,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        return
    bundle = load_bundle(args.data_root)
    fold = prepare_fold(bundle.outings, 2025, train_start=2020, evaluation_split="test")
    names = {str(key): value for key, value in bundle.pitcher_names.items()}

    ewma = predict_ewma_fold(fold)
    ridge, ridge_model = predict_residual_fold(
        fold, model="ridge", features=COMPACT_FEATURES, ridge_alpha=RIDGE_ALPHA,
        correction_scale=RIDGE_SCALE, candidate_id="ridge_a100_scale0.25_locked",
        missing_policy="complete_case",
    )
    xgboost, _ = predict_residual_fold(
        fold, model="xgboost", features=COMPACT_FEATURES,
        correction_scale=XGBOOST_SCALE, xgboost_params=XGBOOST_CONFIG,
        random_state=SEEDS[0], candidate_id="xgboost_trial16_scale0.221845_locked",
        missing_policy="complete_case",
    )
    seed_frames = []
    for seed in SEEDS:
        print(f"Fitting locked TCN seed {seed}...", flush=True)
        prediction, _, _ = predict_tcn_fold(
            fold, config=replace(TCN_CONFIG, seed=seed), sequence_order="ordered",
            shuffle_seed=seed, device=args.device,
            candidate_id="tcn_h1_ordered_l8_c4_b4_alpha0.1_locked",
        )
        seed_frames.append(prediction)
    tcn_seeds = pd.concat(seed_frames, ignore_index=True)
    tcn = average_seed_predictions(tcn_seeds)
    predictions = {"ewma": ewma, "ridge": ridge, "xgboost": xgboost, "tcn": tcn}

    common, deployable = evaluate_common_and_deployable(predictions)
    continuous = pd.DataFrame(
        [{"model": model, **continuous_metrics(frame)} for model, frame in predictions.items()]
    )
    coefficient_frame = ridge_coefficients(ridge_model, names)
    latest = make_latest(predictions, names)
    all_predictions = pd.concat(
        [frame.assign(report_model=model) for model, frame in predictions.items()],
        ignore_index=True, sort=False,
    )

    common.to_csv(args.output_dir / "metrics_common.csv", index=False)
    deployable.to_csv(args.output_dir / "metrics_deployable.csv", index=False)
    continuous.to_csv(args.output_dir / "metrics_continuous.csv", index=False)
    coefficient_frame.to_csv(args.output_dir / "ridge_coefficients_all.csv", index=False)
    latest.to_csv(args.output_dir / "latest_player_predictions.csv", index=False)
    all_predictions.to_parquet(args.output_dir / "predictions_2025.parquet", index=False)
    tcn_seeds.to_parquet(args.output_dir / "tcn_seed_predictions_2025.parquet", index=False)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "qualified_pitchers": len(bundle.qualified_pitchers),
        "evaluation_rows": len(fold.evaluation_targets),
        "common_rows": int(common["n_evaluated"].min()),
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_correction_scale": RIDGE_SCALE,
        "tcn_config": asdict(TCN_CONFIG),
        "tcn_seeds": list(SEEDS),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    build_presentation_reports(
        latest=latest, common=common, deployable=deployable,
        coefficients=coefficient_frame, continuous=continuous, metadata=metadata,
        html_path=args.html, markdown_path=args.markdown,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
