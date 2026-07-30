# MLB 선발투수 다음 경기 Stuff+ 범주 예측

19명의 MLB 선발투수를 대상으로 다음 공식 선발 경기의 Stuff+ 범주
`Low(0) / Middle(1) / High(2)`를 예측하는 시간 순서 모델링 저장소입니다.
범주 경계는 투수별·fold별 학습 구간 Stuff+의 q33/q67이며, `q67`과 같은
값은 Middle로 분류합니다.

이 문서에서 **사전 확정 설정**은 2022~2024 검증 결과만 사용해 모델,
피처, 하이퍼파라미터, random seed와 앙상블 비율을 결정하고 파일로
저장했다는 뜻입니다. 2025 결과를 본 뒤 이 설정을 바꾸지 않습니다.
파일명은 `locked_model_config.json`이지만, 아래에서는 직관적으로
“사전 확정 설정 파일”이라고 부릅니다. 이것은 데이터 누수와 같은 말이
아니며, 2025를 이용한 사후 조정을 막는 절차입니다.

## 최종 결과물에 실제 포함된 모델

단순히 기존 피처와 EWMA4만으로 예측한 결과가 아닙니다. 정상 selection과
사전 확정 설정으로 수행한 2025 final에는 아래 여섯 출력이 모두 실제
학습·평가되어 있습니다.

| 최종 출력 | 모델 입력과 역할 |
|---|---|
| `ewma` | 최근 Stuff+의 EWMA4만 사용하는 기준선 |
| `ridge` | compact16으로 EWMA4 residual을 예측 |
| `xgboost` | compact16으로 EWMA4 residual을 예측 |
| `ordinal_linear` | compact16 + fold-local EWMA tertile position으로 Low/Middle/High를 직접 예측 |
| `tcn_h1_ordered` | 최근 8경기의 raw outing 15차원 시퀀스를 causal TCN으로 인코딩하고 투수별 H1 bias로 residual을 예측 |
| `ridge_tcn_ensemble` | 검증 단계에서 확정한 beta로 Ridge와 TCN residual correction을 결합 |

### 6개 모델의 최종 2025 성능

`Common`은 여섯 모델이 모두 예측 가능한 동일 497경기에서의 공정 비교이고,
`Deployable`은 각 모델이 실제 예측 가능한 전체 범위입니다.

| 모델 | Common N | Common accuracy | Common balanced accuracy | Common macro-F1 | Common ordinal MAE | Deployable N / 560 | Coverage | Deployable accuracy | Deployable balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EWMA4 | 497 | 0.490946 | 0.474060 | 0.478178 | 0.569416 | 560 | 1.0000 | 0.501786 | 0.487629 |
| Ridge | 497 | 0.498994 | 0.482530 | 0.487167 | **0.557344** | 497 | 0.8875 | 0.498994 | 0.482530 |
| XGBoost | 497 | **0.501006** | **0.484720** | **0.489101** | 0.559356 | 497 | 0.8875 | 0.501006 | 0.484720 |
| Ordinal linear | 497 | 0.474849 | 0.461470 | 0.446586 | 0.661972 | 560 | 1.0000 | 0.487500 | 0.479667 |
| TCN H1-4 | 497 | 0.498994 | 0.482675 | 0.485985 | 0.561368 | 560 | 1.0000 | **0.512500** | **0.498267** |
| Ridge–TCN ensemble, beta=0 | 497 | 0.498994 | 0.482530 | 0.487167 | **0.557344** | 497 | 0.8875 | 0.498994 | 0.482530 |

주요 모델 우열은 표본이 같은 Common 열로 판단합니다. Deployable 열은
coverage와 함께 해석해야 합니다. 사전 확정된 ensemble beta가 0이므로
앙상블의 최종 결과가 Ridge와 같은 것은 의도된 결과입니다.

TCN에는 compact16 이동평균·slope 피처가 들어가지 않습니다. 각 target
이전 최근 경기의 Stuff+, 투구 수, 휴식, 주력 패스트볼 물리량, 릴리스,
구종 비율로 구성한 raw sequence를 사용합니다.

검증 단계에서 확정된 TCN은 `H1 ordered, L=8, channels=4, bottleneck=4,
alpha=0.1`이며 5개 seed로 2020~2024를 재학습했습니다.
`tcn_predictions_2025.parquet`에는 `5 seeds × 560경기 = 2,800행`이,
`final_2025_predictions.parquet`에는 seed 평균 TCN 예측이 저장됩니다.

## 용어를 엄밀히 구분

이 저장소에서 “공통”과 “거리”는 각각 다음 의미입니다.

- **공통 tabular 피처**: Ridge와 XGBoost가 동일하게 사용하는 기존
  compact16입니다. Ordinal은 여기에 `ewma_tertile_position` 하나를
  추가합니다.
- **공유 TCN encoder**: 19개 투수의 sequence를 하나의 causal TCN
  encoder가 함께 학습합니다. 투수마다 별도의 TCN을 만드는 구조가
  아닙니다.
- **공통 평가 표본(common sample)**: 모든 비교 모델이 예측 가능한
  `row_id` 교집합입니다. 2025에는 497경기입니다.
- **모델별 가용 표본(deployable sample)**: 각 모델이 실제 예측 가능한
  전체 target입니다. EWMA·ordinal·TCN은 560경기, compact16
  Ridge·XGBoost는 497경기입니다.
- **Ordinal distance loss**: 예측 확률과 실제 범주 사이의 순서상 거리를
  손실로 주는 후보입니다. 구현하고 평가했지만 최종 선택 기준에서는
  distance weight=0을 최종 설정으로 확정했습니다.
- **Ordinal MAE**: 모든 모델에 보고하는 평가 지표입니다. distance loss를
  최종 채택하지 않았더라도 Low↔High 오류를 Low↔Middle보다 크게
  측정합니다.

따라서 모든 모델에 같은 피처를 억지로 넣은 것이 아닙니다. Tabular
summary와 TCN raw sequence를 의도적으로 분리하고, 공정 비교는 동일
common row에서 수행합니다.

## 모델별 입력 피처

### Compact16: Ridge와 XGBoost의 공통 피처

```text
prior_minus_ewma4
mean5_minus_ewma4
stuff_plus_slope_last5
prev_start_pitch_count
rest_days
workload_density_3starts
release_speed_slope5
release_spin_rate_slope5
release_extension_slope5
release_pos_x_slope5
release_pos_z_slope5
arm_angle_slope5
pfx_x_slope5
pfx_z_slope5
breaking_share_slope5
offspeed_share_slope5
```

Ridge와 XGBoost는 현재 Stuff+를 직접 회귀하지 않고 다음 residual을
학습합니다.

```text
residual = actual Stuff+ - EWMA4
predicted Stuff+ = EWMA4 + correction_scale × predicted residual
```

### Ordinal linear 입력

Ordinal linear는 compact16에 다음 training-fold-only 위치값을 더합니다.

```text
ewma_tertile_position
  = (EWMA4 - q33) / (q67 - q33 + epsilon)
```

`EWMA4-q33`과 `EWMA4-q67`을 동시에 넣지 않습니다.

### TCN raw15 sequence 입력

TCN은 compact16을 사용하지 않고 target 이전의 최근 L개 outing에서
다음 원시 집계값을 받습니다.

```text
stuff_plus
pitch_count
rest_days_log
long_gap
season_start
primary_fb_velocity
primary_fb_spin
primary_fb_pfx_x
primary_fb_pfx_z
release_extension
release_pos_x
release_pos_z
arm_angle
fastball_share
breaking_share
```

TCN 입력에서 이동평균, slope5, `prior_minus_ewma4`,
`mean5_minus_ewma4`, 전체 구종 평균 구속은 제외합니다.
`valid_timestep`, `primary_fb_available`, `stuff_plus_available`은 결측과
padding을 구분하는 mask로 별도 전달합니다.

## 검토한 모델과 수식

### Ordinal proportional-odds 모델

```text
score_i,t = beta_i^T x_i,t
theta_2,i = theta_1,i + softplus(delta_i) + epsilon
q1 = P(Y >= 1) = sigmoid(score - theta_1)
q2 = P(Y >= 2) = sigmoid(score - theta_2)
P(Low) = 1 - q1
P(Middle) = q1 - q2
P(High) = q2
```

기본 cumulative BCE와 다음 거리 손실 후보를 함께 검토했습니다.

```text
L_distance = sum_c P(Y=c) × |c - true_class|
```

검토 grid:

- L2: `{0.1, 1.0, 10.0, 100.0}`
- distance weight: `{0.0, 0.1, 0.25}`
- deterministic LBFGS

거리 후보는 실제로 구현·평가됐습니다. 최종에서 빠진 이유는 아래
validation 표처럼 “거리 지표는 개선했지만 주 선택 점수가 낮았기 때문”이며,
고려하지 않은 것이 아닙니다.

| Ordinal 후보 | Mean balanced accuracy | Std | Selection score | Ordinal MAE | 평균 extreme errors |
|---|---:|---:|---:|---:|---:|
| distance=0 | **0.473438** | 0.044491 | **0.451193** | 0.645967 | 80.33 |
| distance=0.25 | 0.465934 | **0.038318** | 0.446775 | **0.634932** | **66.67** |
| distance=0.1 | 0.467256 | 0.047074 | 0.443719 | 0.644717 | 74.33 |

즉 distance=0.25는 ordinal MAE와 극단 오류 수에서는 더 좋았습니다.
하지만 사전에 정한 selection score는 balanced accuracy와 안정성을
우선했기 때문에 distance=0이 사전 확정 설정 파일에 저장됐습니다.

### Shared TCN

공유 encoder 뒤에 다음 세 pitcher-head를 비교했습니다.

```text
H0: r_hat = global_head(h)
H1: r_hat = global_head(h) + pitcher_bias_i
H2: r_hat = global_head(h) + pitcher_vector_i^T h + pitcher_bias_i
```

최종 Stuff+:

```text
predicted Stuff+ = EWMA4 + alpha × r_hat
```

구조와 학습 후보:

- causal Conv1d kernel=2, dilations=`[1,2,4]`
- sequence length `{5,8}`
- internal channels `{4,8}`
- bottleneck `{4,8}`
- head `{H0,H1,H2}`
- ordered와 within-window shuffled
- alpha `{0.1,0.25,0.5}`
- ordinal auxiliary weight `{0,0.25,0.5}`
- Huber residual loss
- 투수별 loss를 먼저 평균한 뒤 19명을 동일 가중 평균
- seeds `20260722, 20260723, 20260724, 20260725, 20260726`

필수 ablation은 L=8에서 비교했고, ordered H1 기준모델을 L=5에서도
평가했습니다. 선택된 작은 TCN-4의 결과:

| 후보 | Mean BA | Seed-year std | Selection score | Seed std | Params |
|---|---:|---:|---:|---:|---:|
| TCN H1, channels=4, L=8 | **0.554810** | 0.026263 | **0.541679** | 0.002492 | 248 |
| H0 global-only | 0.553044 | 0.024172 | 0.540958 | 0.002256 | 641 |
| H1 + ordinal auxiliary 0.25 | 0.552638 | 0.023818 | 0.540730 | 0.002792 | 668 |
| H1 shuffled | 0.550494 | **0.020134** | 0.540427 | 0.002804 | 660 |
| H2 pitcher-vector | 0.553379 | 0.025995 | 0.540382 | **0.001485** | 736 |
| H1 ordered, channels=8, L=8 | 0.551921 | 0.023263 | 0.540289 | 0.002011 | 660 |
| H1 ordered, channels=8, L=5 | 0.551519 | 0.023973 | 0.539532 | 0.003187 | 660 |

H2는 더 복잡하지만 H1/TCN-4보다 selection score가 높지 않아
채택하지 않았습니다. Ordered가 shuffled보다 조금 높았지만 차이는 작아
시간 순서 정보의 추가 이득도 제한적이었습니다.

현재 기본 실행 경로는 다음처럼 분리되어 있습니다.

```text
원자료 수집/로딩
  → deterministic fold_manifest
  → 2022·2023·2024 rolling selection
  → 모델·피처·파라미터를 locked_model_config.json에 사전 확정
  → 2020~2024 refit
  → 사전 확정 설정을 그대로 사용해 2025 최종 평가 1회
```

`select_models.py`는 2025 성적값을 읽어 후보를 평가하지 않습니다.
`final_evaluate.py`는 사전 확정 설정 파일의 코드·데이터·manifest
fingerprint를 검증한 뒤 확정된 후보만 재학습합니다. Optuna, 후보 정렬,
재선택, correction-scale 조정은 수행하지 않습니다.

## 실제 완료 결과

- 정상 selection: 2026-07-30 완료, 59개 후보 요약, 2025 outcome 사용 0건
- 최종 2025 평가: 19명, target-eligible 560경기
- 공통 비교 표본: 모든 모델이 함께 예측 가능한 497경기. 이는
  Ridge/XGBoost의 compact16 결측 때문에 정해진 교집합이며,
  TCN·EWMA·ordinal은 별도로 560경기 모두 예측
- 2022~2024 최고 selection score: XGBoost `0.552473`
- 2025 common accuracy: XGBoost `50.10%`, Ridge/TCN `49.90%`,
  EWMA4 `49.09%`, ordinal linear `47.48%`
- 2025 deployable accuracy/coverage: TCN `51.25% / 100%`,
  XGBoost `50.10% / 88.75%`
- 신규 모델 replacement gate: 통과하지 않음

자세한 모델 수식, 검증표, 2025 결과와 한계는
[FINAL_MODEL_REPORT.md](FINAL_MODEL_REPORT.md)에 정리했습니다. 기존
PDF와의 수치 비교 및 원인 분석은 README에 섞지 않고
[LEGACY_VS_CURRENT_COMPARISON_REPORT.md](LEGACY_VS_CURRENT_COMPARISON_REPORT.md)에
분리했습니다.

## 선택 규칙과 채택·비채택 결과

TCN은 단일 최고 seed를 고르지 않습니다.

```text
TCN selection score
  = 5 seeds × 3 validation years의 balanced accuracy 평균
  - 0.5 × 같은 15개 값의 표준편차
```

비-seeded 모델은 2022·2023·2024 세 값에 같은 식을 적용합니다.
최고 점수와 0.003 이내인 후보는 parameter count가 작은 쪽을 선택합니다.
Macro-F1, ordinal MAE, High recall, extreme error, seed std, parameter count,
coverage도 함께 보고하지만 사전 확정 후보의 기본 순위는 위 식을
따릅니다.

기존 residual 후보도 같은 2022~2024 fold에서 다시 비교했습니다.

- Ridge alpha `{1,10,30,100}`
- Ridge correction scale `{0.25,0.5,1.0}`
- XGBoost staged TPE `32 candidates → top 10 → top 3`
- TCN 선택 이후 ensemble beta `{0,0.1,0.2,0.3}`

| Family별 사전 확정 후보 | Mean BA | Std | Selection score | 채택 상태 |
|---|---:|---:|---:|---|
| XGBoost trial16, scale=0.221845 | **0.561329** | **0.017712** | **0.552473** | 기존 주 모델 유지 |
| EWMA4 | 0.558300 | 0.023146 | 0.546727 | 기준선 유지 |
| Ridge alpha=100, scale=0.25 | 0.553851 | 0.023795 | 0.541954 | 기존 해석 모델 유지 |
| TCN H1-4, L=8, alpha=0.1 | 0.554810 | 0.026263 | 0.541679 | 진단 모델로 저장, 기존 모델 대체 실패 |
| Ordinal L2=0.1, distance=0 | 0.473438 | 0.044491 | 0.451193 | 진단 모델로 저장, 기존 모델 대체 실패 |

`replacement_gate_passed=false`는 TCN이나 ordinal을 계산하지 않았다는
뜻이 아니라, 실제 계산했지만 기존 모델의 주 역할을 자동으로
대체시키지 않았다는 뜻입니다.

TCN 선택 뒤에만 Ridge–TCN residual ensemble을 평가했습니다.

| beta | Mean BA | Selection score | Params |
|---:|---:|---:|---:|
| 0.0 | 0.553851 | 0.541954 | 323 |
| 0.1 | 0.553387 | 0.541959 | 571 |
| 0.2 | 0.555065 | 0.543077 | 571 |
| 0.3 | **0.556742** | **0.544192** | 571 |

beta=0.3의 raw score가 가장 높았지만 beta=0과 차이가 0.002238로
0.003 simplicity margin 안이어서 beta=0을 최종 설정으로 확정했습니다.
따라서 최종 ensemble 파일은 의도적으로 Ridge와 동일합니다.

## 사전 확정 설정으로 수행한 2025 결과

### Common 497경기

| 모델 | Accuracy | Balanced accuracy | Macro-F1 | Ordinal MAE | Extreme errors |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.501006** | **0.484720** | **0.489101** | 0.559356 | 30 |
| Ridge | 0.498994 | 0.482530 | 0.487167 | **0.557344** | **28** |
| TCN H1-4 | 0.498994 | 0.482675 | 0.485985 | 0.561368 | 30 |
| EWMA4 | 0.490946 | 0.474060 | 0.478178 | 0.569416 | 30 |
| Ordinal linear | 0.474849 | 0.461470 | 0.446586 | 0.661972 | 68 |
| Ridge–TCN ensemble, beta=0 | 0.498994 | 0.482530 | 0.487167 | **0.557344** | **28** |

### Deployable 560 target rows

| 모델 | Predicted / eligible | Coverage | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| TCN H1-4 | 560 / 560 | **1.0000** | **0.512500** | **0.498267** |
| EWMA4 | 560 / 560 | **1.0000** | 0.501786 | 0.487629 |
| XGBoost | 497 / 560 | 0.8875 | 0.501006 | 0.484720 |
| Ridge | 497 / 560 | 0.8875 | 0.498994 | 0.482530 |
| Ordinal linear | 560 / 560 | **1.0000** | 0.487500 | 0.479667 |
| Ridge–TCN ensemble, beta=0 | 497 / 560 | 0.8875 | 0.498994 | 0.482530 |

TCN의 560행 성능과 XGBoost/Ridge의 497행 성능은 표본이 다르므로
deployable accuracy만으로 직접 우열을 정하지 않습니다. 모델 선택과
공정 비교는 common 표본을 사용하고 coverage는 별도 판단합니다.

## 설치와 테스트

Python 3.12 환경에서 실행했습니다.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

현재 전체 테스트 결과는 `89 passed, 2 skipped`입니다.

빠른 end-to-end 확인은 축소 프로필을 명시해서 실행합니다.

```powershell
python select_models.py `
  --demo `
  --smoke-test `
  --output-dir experiments/runs/demo_selection

python final_evaluate.py `
  --demo `
  --locked-config experiments/runs/demo_selection/locked_model_config.json `
  --output-dir experiments/runs/demo_final_2025
```

## 원자료 수집 CLI

저장소에는 Statcast와 FanGraphs game-log 수집 코드가 포함되어 있습니다.
네트워크 호출이므로 날짜와 대상 투수 ID를 명시해야 합니다.

```powershell
python -m lib.collect_pitcher_statcast `
  --pitcher-ids 450203 542881 543243 554430 571578 592332 592791 605135 607536 608379 621244 622491 656302 656605 657277 663903 664285 668678 669302 `
  --start 2020-01-01 `
  --end 2025-12-31 `
  --output-dir data/statcast_mlb_stable_starters_2020_2025

python -m lib.collect_fangraphs_stuff `
  --outings data/stable_starter_pitcher_seed_2020_2025.parquet `
  --output data/fangraphs_stuff_mlb_stable_starters_2020_2025.parquet `
  --start-year 2020 `
  --end-year 2025
```

현재 재현 실행은 이미 연도별로 분리된 다음 입력을 사용합니다.

- `data/statcast_mlb_stable_starters_2020/`
- `data/statcast_mlb_stable_starters_2021_2025/`
- `data/fangraphs_stuff_mlb_stable_starters_2020.parquet`
- `data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet`
- `data/mlb_official_pitching_2021_2025.parquet`

원 Statcast에 신뢰할 수 있는 행 단위 선발 여부가 없으면 builder가
`StarterInferenceWarning`을 냅니다. 가능한 경우
`--starter-status <row-level file>`을 함께 전달하십시오.

## 정상 모델 선택

```powershell
python select_models.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/mlb_official_pitching_2021_2025.parquet `
  --output-dir experiments/runs/unified_selection
```

기본 정상 프로필은 TCN seeds
`20260722, 20260723, 20260724, 20260725, 20260726`, 250 epochs,
XGBoost TPE 32 trials를 사용합니다. TCN 구조 ablation은 L=8에서
사전 고정하고, ordered H1 기준모델만 L=5를 추가해 두 sequence length를
비교합니다.

주요 selection 산출물:

- `fold_manifest.parquet`
- `locked_model_config.json`
- `validation_oof_predictions.parquet`
- `validation_metrics_by_year.csv`
- `validation_metrics_summary.csv`
- `validation_metrics_by_seed.csv`
- `feature_coverage.csv`
- `primary_fastball_mapping_by_fold.csv`

## 사전 확정 설정으로 수행하는 2025 최종 평가

Selection이 성공해 `locked_model_config.json`에 모델·피처·파라미터가
확정된 뒤에만 실행합니다.

```powershell
python final_evaluate.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/mlb_official_pitching_2021_2025.parquet `
  --locked-config experiments/runs/unified_selection/locked_model_config.json `
  --output-dir experiments/runs/unified_final_2025
```

주요 최종 산출물:

- `final_2025_predictions.parquet`
- `final_2025_common_sample_metrics.csv`
- `final_2025_deployable_metrics.csv`
- `final_2025_per_pitcher_metrics.csv`
- `ordinal_probabilities_2025.parquet`
- `tcn_predictions_2025.parquet`
- `model_agreement_2025.parquet`
- `confusion_matrix_<model>.csv`
- `extreme_errors_<model>.csv`

## 모델과 데이터 계약

### History와 target

- `history_eligible`: 공식 선발이며 이후 경기의 과거 정보로 사용할 수 있는
  outing입니다.
- `target_eligible`: 공식 선발, 50구 이상, Stuff+ target 존재 조건을 모두
  만족하는 outing입니다.
- 50구 미만 공식 선발은 target에서는 제외하지만 다음 경기 history에는
  남습니다.
- history, EWMA, rest는 `pitcher, year`가 아니라 투수의 전체 시간축에서
  계산합니다.
- 연도 전환은 `season_start`, 21일 초과 공백은 `long_gap`으로 표시하고
  `rest_days_capped=min(rest_days,30)`,
  `rest_days_log=log1p(rest_days_capped)`를 사용합니다.

실제 audit row:

- starter/Stuff audit 2,993행
- 공식 starter history/Stuff 2,992행
- 전체 기간 50구 이상 target 2,947행
- 중복 row_id 0
- 2025 target 560행

### 구종 분류와 주력 패스트볼

```text
fastball share types: FF, SI, FC, FA
primary fastball candidates: FF, SI, FA
breaking: SL, ST, CU, KC, SV, CS
offspeed: CH, FS, FO, SC
```

FC는 fastball share에는 포함되지만 기본 primary 후보에서는 제외됩니다.
Share denominator는 알려진 fastball+breaking+offspeed pitch만 사용하고,
unknown 비중은 `known_pitch_share`로 보존합니다.

FF/SI/FA별 outing count, velocity, spin, pfx_x, pfx_z를 저장합니다.
각 fold에서 해당 투수의 학습 구간 usage가 가장 많은 유형만
`primary_fb_type`으로 정한 뒤 generic primary 물리량을 만듭니다.
Validation/test usage로 primary type을 다시 바꾸지 않습니다.

### 결측 처리

- Ridge/XGBoost: 실제 compact16에 대한 complete case, Ridge scaler는
  training에서만 적합
- Ordinal: 투수별 training median/scaler, 없으면 pooled training fallback
- TCN: 투수별 training median/scaler, 없으면 pooled training fallback,
  availability와 valid-timestep mask 유지

2025에서 primary-fastball 물리량은 545/560행에 직접 관측되지만 TCN은
training-only imputation/mask 정책으로 560행 모두 예측합니다.
Compact16의 `workload_density_3starts` 결측 63행 때문에 Ridge/XGBoost
coverage는 497/560=0.8875입니다.

### Fold와 누출 방지

| Fold | 학습 | 평가 | selection 사용 |
|---:|---|---|---|
| 2022 | 2020~2021 | 2022 | 예 |
| 2023 | 2020~2022 | 2023 | 예 |
| 2024 | 2020~2023 | 2024 | 예 |
| 2025 | 2020~2024 | 2025 | 아니오, 설정 확정 이후 final 1회 |

`row_id`는 canonical `[pitcher, game_pk, game_date]`의 SHA-256입니다.
q33/q67, primary type, imputer, scaler는 각 fold 학습 구간에서만
계산합니다. `select_models.py`의 development 최대 연도는 2024이며
selection OOF에 2025 outcome은 0건입니다.

2021~2025 매년 target-eligible 공식 선발 20경기 이상이라는 historical
roster rule은 19명 대상 명단을 정하는 데 사용됩니다. 여기서 2025는
roster availability에만 쓰며 Stuff+ outcome을 모델 점수에 사용하지
않습니다.

사전 확정 설정 파일이 검증하는 값:

```text
dataset:
db4cca7577b0758ca1f1e3b929f68a6186a47383a8cb1866aef482eb13695604

fold manifest:
347c4845323204d7ccc12d0212f9844a948ea9aad6b04c7ac9d525dd913f550d

code:
7eb80cd95ebf13480c8a38dbbe74a5af752bf58c+analysis-sha256:c11fd0ae3f34f4b8b5db0422eeb4c9fef23f44e2e12ee0fb0fe31344d58ba053
```

Final은 이 값이 하나라도 다르면 실행을 거부합니다. Final 모듈은 candidate
생성, Optuna, 검증 순위 정렬, 2025 기반 correction-scale 변경을 하지
않습니다.

### 검증 단계에서 사전 확정된 구체적 파라미터

- EWMA span=4
- Ridge alpha=100, correction scale=0.25
- XGBoost trial=16, correction scale=0.2218450676
- XGBoost trees=100, max depth=2, learning rate=0.0661012188
- XGBoost min child weight=13.9429000
- XGBoost reg alpha=0.6152831, reg lambda=29.1417558
- XGBoost subsample=0.6825797, column sample=0.7955172
- Ordinal L2=0.1, distance weight=0, LBFGS max_iter=200
- TCN H1 ordered, L=8, channels=4, bottleneck=4, alpha=0.1,
  dropout=0.1, learning rate=0.001, epochs=250
- Ensemble beta=0

## 평가 지표와 진단

- Accuracy: 전체 정답률
- Balanced accuracy: 실제로 존재하는 Low/Middle/High recall의 평균
- Macro-F1: Low/Middle/High 고정 3-class F1 평균
- Ordinal MAE: `mean(abs(predicted_class-true_class))`
- Extreme error: Low→High 또는 High→Low
- Coverage: `n_predicted / n_total_eligible`

Ridge와 TCN의 사전 확정 2025 진단:

- 공통 deployable 497행
- 범주 agreement 96.7807%
- disagreement 16행
- residual correlation 0.325743
- disagreement에서 Ridge와 TCN accuracy 모두 43.75%
- Ridge 오류를 TCN이 수정한 수 7
- TCN이 새로 만든 extreme error 2

이 진단은 2025 이후 모델을 다시 고르는 데 사용하지 않았습니다.

## 자동 검증 범위

현재 결과는 `89 passed, 2 skipped`입니다. 두 skip은 PyTorch가 설치된
환경에서 의도적으로 건너뛰는 “PyTorch 미설치 오류 경로” 테스트입니다.

검증 항목:

- 모든 sequence game date가 target game date보다 과거인지 확인
- fold q33/q67, imputer, scaler, primary type의 validation 누출 차단
- validation에서 SI usage가 커도 training FF primary가 유지되는지 확인
- 40구 공식 선발이 target에서 빠지고 다음 history에는 남는지 확인
- known pitch share 합과 TCN raw15 계약 확인
- q67 tie가 Middle인지 확인
- common row_id 교집합이 모든 후보에서 동일한지 확인
- 여러 fold의 q33/q67을 행별로 적용하는지 확인
- TCN seed 불안정성이 selection score에 반영되는지 확인
- beta=0 ensemble이 Ridge와 정확히 같은지 확인
- 사전 확정 설정 파일 없음, fingerprint/code 불일치 시 final 실패 확인
- final이 selection 함수를 호출하지 않는지 확인
- ordinal 확률 비음수, 합=1, argmax 일치 확인
- legacy wrapper가 selection/final 책임을 다시 섞지 않는지 확인

## Legacy 진입점

이전 파일명은 thin wrapper로 유지됩니다.

- `search_xgboost_fast.py`: 통합 selection 경로로 전달하며 2025를 평가하지 않음
- `stuff_mlb_temporal_final.py`: 사전 확정 설정 파일을 필수로 받아 통합
  final 경로로 전달

과거 코드의 정확한 재현 산출물은
`experiments/runs/legacy_snapshot_xgboost/`,
`experiments/runs/legacy_snapshot_reproduction/`,
`experiments/runs/legacy_reproduction/final/`에 보존됩니다.

과거 성능 수치와 현재 결과의 비교는 이 README에 넣지 않았습니다.
동일 497경기의 경계값·정답 범주·연속 예측을 분해한 결과는
[LEGACY_VS_CURRENT_COMPARISON_REPORT.md](LEGACY_VS_CURRENT_COMPARISON_REPORT.md)에
있습니다.

## 알려진 한계

- 원 Statcast에 신뢰할 수 있는 행 단위 starter 파일이 없으면
  `StarterInferenceWarning`을 냅니다. 가능한 경우 `--starter-status`를
  제공해야 합니다.
- 2025-05-17 Charlie Morton의 GS=0 relief audit row는
  `history_eligible=false`, `target_eligible=false`로 scoring에서
  제외했습니다.
- 선택된 TCN은 residual-only이며 ordinal auxiliary weight=0입니다.
  따라서 TCN 파일의 probability 열은 스키마상 존재하지만 결측이고,
  직접 ordinal 확률은 `ordinal_probabilities_2025.parquet`에 있습니다.
- 2025는 최종 진단 한 시즌입니다. 이 결과로 모델이나 파라미터를 다시
  선택하지 않았습니다.
- `FINAL_MODEL_REPORT.html`은 과거 legacy 스냅샷입니다. 최신 통합 결과는
  `FINAL_MODEL_REPORT.md`에, 기존 PDF와의 비교는 별도 비교 보고서에
  기록했습니다.
