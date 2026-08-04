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
        build_reports(
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
    build_reports(
        latest=latest, common=common, deployable=deployable,
        coefficients=coefficient_frame, continuous=continuous, metadata=metadata,
        html_path=args.html, markdown_path=args.markdown,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
