"""Select and evaluate direct-Stuff+ common/individual hybrid models.

Selection uses only 2022--2024 validation folds.  The selected Ridge,
XGBoost, and TCN configurations are then evaluated once on the fixed 2025
test fold.  All physical inputs for a target outing come from earlier outings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.evaluation import evaluate_common_and_deployable, evaluate_prediction_frame
from lib.hybrid_common_individual import (
    COMMON_SEQUENCE_FEATURES,
    HybridTCNConfig,
    HybridTabularConfig,
    fit_predict_hybrid_tcn,
    predict_hybrid_tabular,
    prepare_hybrid_fold_data,
)
from lib.stuff_experiment import prepare_fold, predict_ewma_fold
from lib.stuff_cli import load_stuff_data


VALIDATION_YEARS = (2022, 2023, 2024)
FINAL_YEAR = 2025
TCN_FINAL_SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/reproducible_mlb_stuff"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/hybrid_direct_ewma_2020_2025"),
    )
    parser.add_argument("--html", type=Path, default=Path("HYBRID_DIRECT_EWMA_REPORT.html"))
    parser.add_argument("--markdown", type=Path, default=Path("HYBRID_DIRECT_EWMA_REPORT.md"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use one candidate per model and one final TCN seed for a smoke run.",
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
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


def candidate_configs(quick: bool):
    ridge = [
        HybridTabularConfig(model="ridge", global_ridge_alpha=g, individual_ridge_alpha=i)
        for g, i in ((30, 100), (100, 100), (100, 300), (300, 300))
    ]
    xgboost = [
        HybridTabularConfig(model="xgboost", common_n_estimators=n, common_max_depth=d,
                            individual_n_estimators=it, individual_max_depth=1)
        for n, d, it in ((75, 1, 15), (100, 2, 20),
                         (150, 2, 25), (100, 3, 20))
    ]
    tcn = [
        HybridTCNConfig(internal_channels=c, bottleneck_dim=b, individual_l2=l2,
                        epochs=200)
        for c, b, l2 in ((4, 4, .05), (4, 4, .10),
                         (8, 4, .10), (4, 8, .50))
    ]
    return (ridge[:1], xgboost[:1], tcn[:1]) if quick else (ridge, xgboost, tcn)


def config_id(model: str, config) -> str:
    raw = asdict(config)
    if model == "ridge":
        return f"ridge_direct_ewma_g{raw['global_ridge_alpha']:g}_i{raw['individual_ridge_alpha']:g}"
    if model == "xgboost":
        return (f"xgb_direct_ewma_n{raw['common_n_estimators']}_d{raw['common_max_depth']}_"
                f"in{raw['individual_n_estimators']}")
    return (f"tcn_direct_ewma_std_warm_c{raw['internal_channels']}_b{raw['bottleneck_dim']}_"
            f"l2{raw['individual_l2']:g}")


def continuous_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(frame["true_stuff_plus"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_stuff_plus"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    error = predicted.loc[valid] - actual.loc[valid]
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "continuous_n": int(valid.sum()),
    }


def score_row(model: str, config, year: int, prediction: pd.DataFrame) -> dict:
    metrics = evaluate_prediction_frame(prediction)
    return {
        "model": model,
        "candidate_id": config_id(model, config),
        "year": year,
        "config_json": json.dumps(asdict(config), sort_keys=True),
        **metrics,
        **continuous_metrics(prediction),
    }


def select_winners(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, candidate), group in validation.groupby(["model", "candidate_id"], sort=True):
        values = pd.to_numeric(group["balanced_accuracy"], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        rows.append({
            "model": model,
            "candidate_id": candidate,
            "mean_balanced_accuracy": mean,
            "std_balanced_accuracy": std,
            "selection_score": mean - .5 * std,
            "mean_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
            "mean_mae": float(pd.to_numeric(group["mae"], errors="coerce").mean()),
            "config_json": group["config_json"].iloc[0],
        })
    summary = pd.DataFrame(rows).sort_values(
        ["model", "selection_score", "mean_balanced_accuracy", "candidate_id"],
        ascending=[True, False, False, True], kind="mergesort",
    )
    summary["selected"] = ~summary.duplicated("model")
    return summary


def average_tcn_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    first = frames[0].copy().sort_values("row_id").reset_index(drop=True)
    for column in ("global_prediction", "global_residual", "pitcher_correction", "predicted_residual", "predicted_stuff_plus"):
        stacked = np.stack([
            frame.sort_values("row_id")[column].to_numpy(float) for frame in frames
        ])
        first[column] = stacked.mean(axis=0)
    from lib.stuff_tabular import classify_stuff_plus_rows
    first["predicted_class"] = classify_stuff_plus_rows(
        first["predicted_stuff_plus"], first["q33"], first["q67"]
    )
    first["candidate_id"] = first["candidate_id"].astype(str) + "_5seed_mean"
    return first


def ridge_coefficients(models: dict, fold_data) -> pd.DataFrame:
    rows = []
    global_model = models["global_model"]
    for name, coefficient in zip(fold_data.common_feature_names, global_model.coef_, strict=True):
        rows.append({"branch": "common", "pitcher": "ALL", "feature": name,
                     "coefficient": float(coefficient)})
    inverse_mapping = {value: key for key, value in fold_data.pitcher_mapping.items()}
    for pitcher_index, model in models["individual_models"].items():
        for name, coefficient in zip(
            fold_data.individual_tabular_feature_names, model.coef_, strict=True
        ):
            rows.append({"branch": "individual", "pitcher": inverse_mapping[pitcher_index],
                         "feature": name, "coefficient": float(coefficient)})
    return pd.DataFrame(rows)


def latest_player_predictions(predictions: dict[str, pd.DataFrame], names: dict[str, str]) -> pd.DataFrame:
    base = predictions["ewma"].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])
    latest = base.sort_values(["pitcher", "game_date", "game_pk"]).groupby("pitcher").tail(1)
    result = latest[["row_id", "pitcher", "game_date", "true_stuff_plus", "true_class", "q33", "q67"]].copy()
    result["pitcher_name"] = result["pitcher"].astype(str).map(names).fillna(result["pitcher"].astype(str))
    for model, frame in predictions.items():
        lookup = frame.set_index("row_id")
        result[f"{model}_prediction"] = result["row_id"].map(lookup["predicted_stuff_plus"])
        result[f"{model}_class"] = result["row_id"].map(lookup["predicted_class"])
    return result.sort_values("pitcher_name").reset_index(drop=True)


def table_html(frame: pd.DataFrame, digits: int = 4) -> str:
    return frame.to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.{digits}f}")


def table_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Small dependency-free Markdown table renderer."""

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered = "" if not np.isfinite(value) else f"{value:.{digits}f}"
            else:
                rendered = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(rendered)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(*, markdown_path: Path, html_path: Path, metadata: dict,
                 selection: pd.DataFrame, common: pd.DataFrame, deployable: pd.DataFrame,
                 continuous: pd.DataFrame, comparison: pd.DataFrame,
                 latest: pd.DataFrame, individual_features: list[str]) -> None:
    selected = selection.loc[selection["selected"], ["model", "candidate_id", "mean_balanced_accuracy", "selection_score"]]
    common_text = "\n".join(f"- `{feature}`" for feature in COMMON_SEQUENCE_FEATURES)
    individual_text = "\n".join(f"- `{feature}`" for feature in individual_features)
    markdown = f"""# 공통·선수별 하이브리드 Stuff+ 예측 실험

생성 시각: {metadata['generated_at']}

## 1. 개요

모든 선수에게 반복되는 상태·구종 구조는 공통 모델로 학습하고, 투구 위치와 구종별 세부 변화 등 나머지 정보는 선수별 보정 모델로 학습했다. 모델 선택은 2022~2024년에만 수행했고 2025년은 마지막 한 번의 고정 평가에 사용했다.

## 2. 데이터와 누수 방지

현재 경기의 Stuff+와 물리 피처는 입력하지 않았다. 각 예측 행은 경기일보다 엄격히 이전인 최대 8경기만 사용하며, 결측치 대치와 정규화 통계도 해당 fold의 훈련 구간에서 선수별로 계산했다.

## 3. 공통 피처

{common_text}

Ridge와 XGBoost는 각 공통 피처의 마지막값·평균·표준편차·추세와 EWMA4(총 53개)를 사용한다. TCN은 정규화된 8경기 시퀀스와 별도의 학습 가능한 EWMA4 입력을 사용한다.

## 4. 선수별 피처

{individual_text}

`release_pos_y`를 포함한 나머지 사용 가능 피처 전체를 과거 시퀀스 요약 및 사전 계산된 lag/slope 형태로 사용한다.

## 5. 모델 구조

- Ridge: EWMA4를 포함한 공통 Ridge 직접 예측 + 선수별 Ridge 보정
- XGBoost: EWMA4를 포함한 공통 부스팅 직접 예측 + 선수별 소형 부스팅 보정
- TCN: 공통 causal TCN 인코더 + 선수별 선형 보정층
- 최종값: 학습된 공통 Stuff+ 예측 + 선수별 보정. EWMA4의 계수도 모델이 학습한다.

계산식은 다음과 같다.

\\[
\\hat y_{{i,t}}=g(C_{{i,t}}, EWMA4_{{i,t}})+h_i(U_{{i,t}})
\\]

Ridge에서는 `g`와 `h_i`가 선형식이고, XGBoost에서는 트리의 합, TCN에서는 causal encoder와 학습 가능한 EWMA skip 및 선수별 선형층이다. 이전 방식의 `EWMA4 + 0.1 × 잔차`는 사용하지 않는다.

## 6. 검증 선택

{table_markdown(selected)}

## 7. 2025 결과

### 공통 표본

{table_markdown(common)}

### 모델별 예측 가능 표본

{table_markdown(deployable)}

### 연속형 Stuff+ 오차

{table_markdown(continuous)}

### 기존 잔차 방식과 직접예측 비교

{table_markdown(comparison)}

## 8. 선수별 마지막 예측

{table_markdown(latest, 2)}

## 9. 해석과 결론

공통 branch는 선수 간 반복되는 변화 패턴을 모으고, 선수별 branch는 같은 변화라도 선수마다 다른 반응을 보정한다. Ridge 계수 전체는 별도 CSV로 남겨 공통 효과와 선수별 효과를 분리해 확인할 수 있게 했다. 성능 판단은 2025 공통 표본의 balanced accuracy와 연속형 MAE를 함께 본다.

## 10. 아키텍처와 파이프라인

```text
Statcast + FanGraphs Stuff+ + MLB 공식 기록
                    │
       경기 단위 집계 / strictly-prior lag
                    │
        선수별 train-only 정규화
             ┌──────┴──────┐
             │             │
       공통 13개       나머지 전체 피처
       ┌─────┼─────┐       │
     Ridge  XGB   TCN      선수별 보정
       └─────┴─────┴───────┘
                    │
     공통 직접 Stuff+ 예측(EWMA 계수 학습) + 개인 보정
                    │
        연속 Stuff+ / 3단계 등급
```
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    css = """body{font-family:Arial,'Noto Sans KR',sans-serif;max-width:1280px;margin:40px auto;padding:0 28px;color:#17202a;line-height:1.65}h1{font-size:34px}h2{margin-top:42px;border-bottom:2px solid #17202a;padding-bottom:8px}code{background:#eef2f5;padding:2px 5px;border-radius:4px}.data{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}.data th,.data td{border:1px solid #d8dee4;padding:7px 9px;text-align:right}.data th{background:#17202a;color:white;position:sticky;top:0}.arch{background:#f3f6f8;border-left:5px solid #e25822;padding:22px;white-space:pre-wrap}ul{columns:2}@media(max-width:800px){ul{columns:1}}"""
    sections = [
        ("1. 개요", "모든 선수의 반복 패턴은 공통 branch에서, 나머지 전체 피처는 선수별 branch에서 학습했다. 2022~2024 검증으로 설정을 고른 뒤 2025를 고정 평가했다."),
        ("2. 데이터와 누수 방지", "현재 경기 입력은 사용하지 않는다. 최대 8개의 strictly-prior 등판만 사용하고 대치·정규화는 훈련 구간에서만 적합한다."),
        ("3. 공통 피처", "<ul>" + "".join(f"<li><code>{html.escape(x)}</code></li>" for x in COMMON_SEQUENCE_FEATURES) + "</ul><p>Ridge/XGBoost: last·mean·std·slope 52개 + EWMA4 1개. TCN: 8경기 시퀀스 + 학습 가능한 EWMA4 입력.</p>"),
        ("4. 선수별 피처", "<ul>" + "".join(f"<li><code>{html.escape(x)}</code></li>" for x in individual_features) + "</ul>"),
        ("5. 모델과 계산식", "<p>세 모델 모두 Stuff+를 직접 예측한다. EWMA4는 고정 기준값이 아니라 계수를 학습하는 공통 피처다.</p><div class='arch'>ŷᵢₜ = g(Cᵢₜ, EWMA4ᵢₜ) + hᵢ(Uᵢₜ)</div><p>Ridge = 공통 선형 직접 예측 + 선수별 선형 보정<br>XGBoost = 공통 트리 직접 예측 + 선수별 소형 트리 보정<br>TCN = 공통 causal encoder + 학습 가능한 EWMA skip + 선수별 선형 보정<br>이전 식 EWMA4 + 0.1 × 잔차는 사용하지 않는다.</p>"),
        ("6. 2022~2024 모델 선택", table_html(selected)),
        ("7. 2025 공통 표본 결과", table_html(common)),
        ("8. 2025 예측 가능 표본", table_html(deployable)),
        ("9. 연속형 결과", table_html(continuous)),
        ("10. 기존 방식과 비교", table_html(comparison)),
        ("11. 선수별 마지막 예측", table_html(latest, 2)),
        ("12. 해석·결론", "직접예측은 모델 간 차이를 실제 예측값에 반영했고 XGBoost와 TCN의 연속 MAE를 개선했다. 반면 3등급 balanced accuracy는 하락했다. 따라서 연속 Stuff+가 목표면 직접예측 XGBoost, 등급 안정성이 목표면 기존 잔차 방식이 유리하다. Ridge 전체 계수는 CSV로 저장했다."),
        ("13. 아키텍처와 파이프라인", "<div class='arch'>Statcast + FanGraphs + MLB 공식 기록\n          ↓\n경기 집계 / strictly-prior lag / train-only 정규화\n          ↓\n공통 13개 + EWMA4 ── Ridge · XGBoost · causal TCN\n나머지 전체 ───────── 선수별 Ridge · XGBoost · 선형 보정\n          ↓\n공통 직접 Stuff+ 예측 + 선수별 보정\n          ↓\n연속 Stuff+ + 3단계 등급</div>"),
    ]
    body = "".join(f"<section><h2>{title}</h2>{content}</section>" for title, content in sections)
    html_path.write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Hybrid Stuff+ Report</title><style>{css}</style></head><body><h1>공통·선수별 하이브리드 Stuff+ 예측</h1><p>{metadata['generated_at']}</p>{body}</body></html>", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.data_root)
    ridge_configs, xgb_configs, tcn_configs = candidate_configs(args.quick)
    configs = {"ridge": ridge_configs, "xgboost": xgb_configs, "tcn": tcn_configs}
    validation_path = args.output_dir / "validation_metrics.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation_rows = validation.to_dict("records")
    else:
        validation = pd.DataFrame()
        validation_rows = []
    if not args.final_only:
        completed = {
            (int(row.year), str(row.model), str(row.candidate_id))
            for row in validation.itertuples()
        } if not validation.empty else set()
        for year in VALIDATION_YEARS:
            print(f"Preparing validation fold {year}...", flush=True)
            fold = prepare_fold(bundle.outings, year, train_start=2020, evaluation_split="validation")
            fold_data = prepare_hybrid_fold_data(fold)
            for model, model_configs in configs.items():
                for config in model_configs:
                    key = (year, model, config_id(model, config))
                    if key in completed:
                        print(f"  cached {year} {key[2]}", flush=True)
                        continue
                    print(f"  {year} {key[2]}", flush=True)
                    if model == "tcn":
                        prediction, _, _ = fit_predict_hybrid_tcn(fold_data, config, device=args.device)
                    else:
                        prediction, _ = predict_hybrid_tabular(fold_data, config)
                    validation_rows.append(score_row(model, config, year, prediction))
                    validation = pd.DataFrame(validation_rows)
                    validation.to_csv(validation_path, index=False)
                    completed.add(key)
    if validation.empty:
        raise FileNotFoundError("No cached validation results; run selection first")
    selection = select_winners(validation)
    selection.to_csv(args.output_dir / "candidate_selection.csv", index=False)
    if args.selection_only:
        print(selection.loc[selection["selected"]].to_string(index=False), flush=True)
        return
    winners = {
        row.model: json.loads(row.config_json)
        for row in selection.loc[selection["selected"]].itertuples()
    }

    print("Preparing fixed 2025 test fold...", flush=True)
    final_fold = prepare_fold(bundle.outings, FINAL_YEAR, train_start=2020, evaluation_split="test")
    final_data = prepare_hybrid_fold_data(final_fold)
    predictions = {"ewma": predict_ewma_fold(final_fold)}
    ridge_prediction, ridge_models = predict_hybrid_tabular(final_data, winners["ridge"])
    xgb_prediction, _ = predict_hybrid_tabular(final_data, winners["xgboost"])
    predictions["hybrid_ridge"] = ridge_prediction
    predictions["hybrid_xgboost"] = xgb_prediction
    tcn_config = HybridTCNConfig(**winners["tcn"])
    seeds = TCN_FINAL_SEEDS[:1] if args.quick else TCN_FINAL_SEEDS
    tcn_frames = []
    for seed in seeds:
        print(f"Fitting final TCN seed {seed}...", flush=True)
        seed_path = args.output_dir / f"tcn_predictions_seed_{seed}.parquet"
        if seed_path.exists():
            frame = pd.read_parquet(seed_path)
        else:
            frame, _, history = fit_predict_hybrid_tcn(
                final_data, replace(tcn_config, seed=seed), device=args.device
            )
            frame.to_parquet(seed_path, index=False)
            history.assign(seed=seed).to_csv(args.output_dir / f"tcn_history_seed_{seed}.csv", index=False)
        tcn_frames.append(frame)
    predictions["hybrid_tcn"] = average_tcn_predictions(tcn_frames)
    common, deployable = evaluate_common_and_deployable(predictions)
    continuous = pd.DataFrame([
        {"model": model, **continuous_metrics(frame)} for model, frame in predictions.items()
    ])
    direct_comparison = common[["model", "accuracy", "balanced_accuracy"]].merge(
        continuous[["model", "mae", "rmse"]], on="model", how="left"
    ).assign(approach="direct_stuff_plus_with_ewma_feature")
    prior_dir = Path("experiments/runs/hybrid_common_individual_2020_2025")
    prior_common_path = prior_dir / "metrics_common.csv"
    prior_continuous_path = prior_dir / "metrics_continuous.csv"
    if prior_common_path.exists() and prior_continuous_path.exists():
        prior_common = pd.read_csv(prior_common_path)
        prior_continuous = pd.read_csv(prior_continuous_path)
        prior_comparison = prior_common[["model", "accuracy", "balanced_accuracy"]].merge(
            prior_continuous[["model", "mae", "rmse"]], on="model", how="left"
        ).assign(approach="ewma_plus_0.1_residual")
        comparison = pd.concat(
            [prior_comparison, direct_comparison], ignore_index=True, sort=False
        )
    else:
        comparison = direct_comparison
    comparison = comparison[
        ["approach", "model", "accuracy", "balanced_accuracy", "mae", "rmse"]
    ]
    names = {str(key): value for key, value in bundle.pitcher_names.items()}
    latest = latest_player_predictions(predictions, names)
    coefficients = ridge_coefficients(ridge_models, final_data)
    all_predictions = pd.concat([
        frame.assign(report_model=model) for model, frame in predictions.items()
    ], ignore_index=True, sort=False)
    actual_features = {
        "common_sequence_features": list(COMMON_SEQUENCE_FEATURES),
        "common_tabular_features": list(final_data.common_feature_names),
        "individual_raw_features": list(final_data.individual_raw_feature_names),
        "individual_derived_features": list(final_data.individual_derived_feature_names),
        "individual_tabular_features": list(final_data.individual_tabular_feature_names),
    }
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "validation_years": list(VALIDATION_YEARS),
        "test_year": FINAL_YEAR,
        "selected_configs": winners,
        "tcn_final_seeds": list(seeds),
        "qualified_pitchers": len(bundle.qualified_pitchers),
        "test_rows": len(final_data.evaluation_targets),
        "common_feature_count": len(COMMON_SEQUENCE_FEATURES),
        "individual_feature_count": len(final_data.individual_tabular_feature_names),
    }
    validation.to_csv(validation_path, index=False)
    common.to_csv(args.output_dir / "metrics_common.csv", index=False)
    deployable.to_csv(args.output_dir / "metrics_deployable.csv", index=False)
    continuous.to_csv(args.output_dir / "metrics_continuous.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_with_residual.csv", index=False)
    latest.to_csv(args.output_dir / "latest_player_predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "hybrid_ridge_coefficients.csv", index=False)
    all_predictions.to_parquet(args.output_dir / "predictions_2025.parquet", index=False)
    (args.output_dir / "actual_feature_split.json").write_text(
        json.dumps(actual_features, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report(
        markdown_path=args.markdown, html_path=args.html, metadata=metadata,
        selection=selection, common=common, deployable=deployable,
        continuous=continuous, comparison=comparison, latest=latest,
        individual_features=(list(final_data.individual_raw_feature_names)
                             + list(final_data.individual_derived_feature_names)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
