# MLB 선발투수 Stuff+ 범주 예측 통합 최종 보고서

- 실행일: 2026-07-30
- 대상: 19명
- 개발/선택 평가: 2022, 2023, 2024 rolling folds
- 최종 refit: 2020~2024
- 사전 확정 설정으로 수행한 최종 평가: 2025
- 정상 selection 실행 시간: 1,191.8초
- 사전 확정 설정 final 실행 시간: 80.0초
- 기존 PDF와의 비교·원인 분석:
  [`LEGACY_VS_CURRENT_COMPARISON_REPORT.md`](LEGACY_VS_CURRENT_COMPARISON_REPORT.md)

이 보고서는 현재 통합 결과만 상세히 기술한다. 2026-07-24 기존 PDF의
수치와 현재 수치의 비교는 위 별도 비교 보고서에 분리했다.

## 1. 결론

모델 선택과 최종 평가를 하나의 사전 확정 설정 경로로 통합했다.
`select_models.py`는 2022~2024만 점수화하고
`locked_model_config.json`에 모델·피처·파라미터·seed를 확정한다.
`final_evaluate.py`는 해당 설정 파일의
코드·데이터·fold fingerprint를 검증한 뒤 2020~2024로 정확히 refit하고,
2025를 한 번 평가했다.

여기서 “사전 확정”은 데이터 누수와 같은 뜻이 아니다. 2022~2024
검증만 보고 설정을 결정한 뒤 2025 결과를 보고 바꾸지 않는 절차이며,
이 절차가 2025를 이용한 사후 튜닝을 방지한다. 파일명과 CLI 옵션에는
기술 이름인 `locked_model_config.json`, `--locked-config`가 남는다.

2022~2024 common-sample selection score가 가장 높은 후보는 XGBoost
`xgboost_tpe16_scale0.221845`였다. 신규 ordinal/TCN은 기존 모델을
대체하기 위한 gate를 통과하지 못했으며, 사전 확정 설정 파일에도
`replacement_gate_passed=false`로 기록되었다. Ridge–TCN ensemble의
최고 raw score는 beta=0.3이었지만 beta=0과의 차이가 0.3%p simplicity
margin 이내였기 때문에 beta=0, 즉 Ridge 그대로를 최종 설정으로
확정했다.

2025 common 497경기에서 accuracy는 XGBoost 50.10%, Ridge와 TCN 49.90%,
EWMA4 49.09%, ordinal linear 47.48%였다. 전체 560개 target row에 대한
deployable 결과에서는 TCN이 100% coverage에서 accuracy 51.25%,
balanced accuracy 49.83%를 기록했다. 그러나 이 2025 결과로 확정 설정이나
모델 역할을 다시 바꾸지 않았다.

## 2. 수정·추가 파일과 역할

| 파일 | 역할 |
|---|---|
| `lib/build_outings.py` | pitch-level Statcast를 공식 outing으로 집계하고 starter 상태를 기록 |
| `lib/stuff_mlb_dataset.py` | Stuff+ 병합, history/target eligibility, cross-season history와 pitch-type 집계 |
| `lib/temporal_splits.py` | SHA-256 row_id, rolling fold, training-only tertile/primary-fastball mapping, fingerprint |
| `lib/evaluation.py` | common/deployable ordinal metrics, coverage, 사전 확정 설정 저장·검증 |
| `lib/preprocessing.py` | fold-training-only 투수별 imputation/scaling과 pooled fallback |
| `lib/ordinal_linear.py` | 투수별 proportional-odds ordinal linear baseline |
| `lib/shared_tcn.py` | raw outing sequence, causal TCN encoder, H0/H1/H2 heads, 손실과 seed 고정 |
| `lib/stuff_experiment.py` | fold 준비와 EWMA/Ridge/XGBoost/ordinal/TCN 예측 adapter |
| `lib/stuff_tabular.py` | compact16 residual regressor와 row-wise tertile 분류 |
| `lib/xgboost_tpe.py` | 2025를 제외한 staged 32→10→3 XGBoost 탐색 |
| `lib/stuff_selection.py` | 2022~2024 OOF 평가, seed 안정성, simplicity rule, 설정 확정 파일 생성 |
| `lib/stuff_final.py` | 사전 확정 설정만 사용한 2020~2024 refit과 2025 최종 산출물 생성 |
| `lib/stuff_cli.py` | raw/prepared/demo 데이터 입력과 qualification 계약 |
| `lib/stuff_code_version.py` | 분석 소스 전체의 정규화 SHA-256 코드 버전 |
| `lib/stuff_demo_data.py` | deterministic end-to-end 테스트 데이터 |
| `build_fold_manifest.py` | 공통 manifest CLI |
| `select_models.py` | 정상 selection CLI |
| `final_evaluate.py` | 사전 확정 설정 final CLI |
| `search_xgboost_fast.py` | 이전 이름을 통합 selection으로 연결하는 thin wrapper |
| `stuff_mlb_temporal_final.py` | 이전 이름을 설정 확정 final로 연결하는 thin wrapper |
| `lib/collect_pitcher_statcast.py` | 투수 ID별 전체 Statcast 수집 |
| `lib/collect_fangraphs_stuff.py` | FanGraphs game-log Stuff+ 수집 |
| `tests/` | leakage, fold, feature, 확률, 설정 파일, wrapper, end-to-end 회귀 테스트 |
| `README.md` | 설치, 수집, selection/final 명령과 출력 계약 |

## 3. 기존 코드에서 확인한 실제 문제와 수정

1. `search_xgboost_fast.py`와 `stuff_mlb_temporal_final.py`가 서로 다른
   validation 범위와 후보 재탐색을 사용했다. 두 이전 진입점은 각각
   통합 selection/final의 thin wrapper로 변경했다.
2. 이전 final 경로는 사전 확정된 XGBoost 설정을 읽지 않고 EWMA span, Ridge,
   scale, feature set, XGBoost를 다시 골랐다. final 모듈은 selection
   함수를 import하지 않으며 사전 확정 설정 파일 없이는 실패하도록 바꿨다.
3. compact 모델이 쓰지 않는 전체 feature의 결측값까지 complete-case로
   요구했다. 각 모델이 실제 사용하는 feature만 검사한다.
4. 50구 미만 공식 선발이 target 필터와 함께 history에서도 사라졌다.
   `history_eligible`과 `target_eligible`을 분리해 50구 미만 선발은 다음
   경기 history에 남겼다.
5. history/rest 계산이 `pitcher, year`에서 끊겼다. 투수 단위의 연속
   시간축으로 바꾸고 `season_start`, `long_gap`, `rest_days_capped`,
   `rest_days_log`를 추가했다.
6. 전체 pitch 평균 구속이 TCN의 패스트볼 물리값처럼 쓰였다. FF/SI/FA
   유형별 집계를 보존하고 fold-training usage로 주력 패스트볼을 정한 뒤
   `primary_fb_velocity/spin/pfx_x/pfx_z`를 만들었다.
7. fastball/breaking/offspeed denominator와 unknown pitch 진단이 불명확했다.
   known pitch만 share denominator로 쓰고 `known_pitch_share`를 보존했다.
8. 여러 fold를 합친 seed 평균과 ensemble에서 같은 투수의 마지막 q33/q67이
   이전 fold 경계를 덮어썼다. 각 row가 가진 fold-local q33/q67을 직접
   사용하도록 수정했다.
9. TCN seed 표준편차가 보고만 되고 선택 점수에는 들어가지 않았다.
   5 seeds × 3 years의 15개 balanced-accuracy 관측치 전체로
   `mean - 0.5 × std`를 계산한다.
10. sequence builder가 target마다 전체 history를 반복 변환해 실제 smoke가
    약 10분 걸렸다. fold history를 한 번 변환한 뒤 index로 sequence를
    조립해 250-epoch 단일 fit을 11.6초로 줄였다.
11. 고정 3-class macro-F1에서 support가 없는 범주가 평균에서 사라질 수
    있었다. Low/Middle/High 고정 label set을 사용한다.
12. 2025-05-17 Charlie Morton의 GS=0 relief row가 starter 추론 audit에
    걸렸다. 이 행은 `history_eligible=false`, `target_eligible=false`로
    유지되어 scoring에는 들어가지 않는다.

## 4. 데이터 계약과 누출 방지

### 4.1 대상과 eligibility

historical roster rule은 2021~2025 매 시즌 target-eligible 공식 선발이
20경기 이상인 19명이다. 2025의 경기 존재 여부는 roster availability로만
사용하며, 2025 Stuff+ 결과는 selection score에 사용하지 않는다.

- starter/Stuff audit row: 2,993
- 공식 starter history/Stuff row: 2,992
- 50구 이상 target row: 2,947
- 중복 row_id: 0
- final 2025 target row: 560
- final common compact16 row: 497

`history_eligible`은 공식 선발 history 사용 가능 여부이고,
`target_eligible`은 공식 선발이면서 pitch count ≥50, Stuff+가 존재하는
경우다. 50구 미만 선발은 target은 아니지만 이후 history에는 포함된다.

원 Statcast에 신뢰할 수 있는 행 단위 starter flag가 없을 때는
`StarterInferenceWarning`을 명시적으로 출력한다. 외부 row-level
`is_starting_pitcher`/GS 자료가 있으면 `--starter-status`로 전달할 수 있다.

### 4.2 row_id, fold와 fingerprint

`row_id`는 canonical `[pitcher, game_pk, YYYY-MM-DD]` JSON을 SHA-256으로
해시한다. q33/q67, 주력 패스트볼, imputer, scaler는 항상 fold 학습
구간만 사용한다.

| 평가 연도 | 학습 연도 | 용도 |
|---:|---|---|
| 2022 | 2020~2021 | selection |
| 2023 | 2020~2022 | selection |
| 2024 | 2020~2023 | selection |
| 2025 | 2020~2024 | 사전 확정 설정 final |

- manifest 행 수: 8,533
- dataset fingerprint:
  `db4cca7577b0758ca1f1e3b929f68a6186a47383a8cb1866aef482eb13695604`
- fold manifest fingerprint:
  `347c4845323204d7ccc12d0212f9844a948ea9aad6b04c7ac9d525dd913f550d`
- code version:
  `7eb80cd95ebf13480c8a38dbbe74a5af752bf58c+analysis-sha256:c11fd0ae3f34f4b8b5db0422eeb4c9fef23f44e2e12ee0fb0fe31344d58ba053`

Final은 이 세 값 중 하나라도 사전 확정 설정 파일과 다르면 실행을 거부한다.

### 4.3 pitch-type 규칙

- fastball share: `FF, SI, FC, FA`
- primary fastball 후보: `FF, SI, FA`
- breaking: `SL, ST, CU, KC, SV, CS`
- offspeed: `CH, FS, FO, SC`
- FC는 기본 share에서 fastball이지만 primary 후보는 아니다.
- share denominator는 fastball+breaking+offspeed known pitch count다.
- unknown/other 비중은 `known_pitch_share`로 진단한다.

2025 final primary type은 2020~2024 usage만으로 결정됐으며, 설정 파일에 투수별
mapping 전체가 저장돼 있다.

## 5. 모델 정의와 사전 확정 파라미터

### 5.1 공통 target

각 투수의 fold-training Stuff+에서 q33/q67을 계산한다.

```text
Low    : value <= q33
Middle : q33 < value <= q67
High   : value > q67
```

### 5.2 EWMA4, Ridge, XGBoost

기본 예측은 최근 Stuff+의 span-4 EWMA다. residual 모델은 다음을 사용한다.

```text
residual = actual Stuff+ - EWMA4
predicted Stuff+ = EWMA4 + correction_scale × predicted residual
```

Ridge와 XGBoost는 기존 compact16 feature 계약을 그대로 사용한다.
검증 단계에서 확정한 설정:

- EWMA span: 4
- Ridge: alpha=100, correction scale=0.25
- XGBoost: trial 16, correction scale=0.2218450676
- XGBoost trees=100, max_depth=2, learning_rate=0.0661012188
- min_child_weight=13.9429000, reg_alpha=0.6152831,
  reg_lambda=29.1417558
- subsample=0.6825797, colsample_bytree=0.7955172,
  gamma=0, hist/max_bin=64

### 5.3 Ordinal linear

입력은 compact16과 하나의 baseline 위치값이다.

```text
ewma_tertile_position = (EWMA4 - q33) / (q67 - q33 + epsilon)
score_i,t = beta_i^T x_i,t
theta_2,i = theta_1,i + softplus(delta_i) + epsilon
q1 = P(Y>=1) = sigmoid(score - theta_1)
q2 = P(Y>=2) = sigmoid(score - theta_2)
P(0)=1-q1, P(1)=q1-q2, P(2)=q2
```

투수별 fold-training StandardScaler와 training-only median/pooled fallback을
쓴다. deterministic LBFGS로 L2 `{0.1,1,10,100}`, distance weight
`{0,0.1,0.25}`를 비교했다. 확정 후보는 L2=0.1, distance weight=0,
max_iter=200이다. ordinal 모델은 Ridge를 자동 대체하지 않는다.

### 5.4 Shared TCN

각 target 이전의 최근 L개 outing만 사용한다.

```text
[PAD, ..., outing_(t-L), ..., outing_(t-1)] → causal TCN → bottleneck h
```

raw15 입력:

1. `stuff_plus`
2. `pitch_count`
3. `rest_days_log`
4. `long_gap`
5. `season_start`
6. `primary_fb_velocity`
7. `primary_fb_spin`
8. `primary_fb_pfx_x`
9. `primary_fb_pfx_z`
10. `release_extension`
11. `release_pos_x`
12. `release_pos_z`
13. `arm_angle`
14. `fastball_share`
15. `breaking_share`

이동평균, slope, EWMA 차이, 전체 pitch 평균 구속은 TCN 입력에 없다.
padding과 실제 0을 구분하는 `valid_timestep`, `primary_fb_available`,
`stuff_plus_available` mask를 별도로 전달한다. 결측값은 투수별
fold-training median, 없으면 pooled training median으로 대체하며 numeric
raw feature는 투수별 training 통계로 scale한다.

encoder는 kernel size 2, left-causal padding, dilation `[1,2,4]`,
GELU, dropout 0.1의 residual Conv1d block 3개다.

```text
H0: r_hat = w_g^T h + b_g
H1: r_hat = w_g^T h + b_g + b_i
H2: r_hat = w_g^T h + b_g + w_i^T h + b_i
Stuff+_hat = EWMA4 + alpha × r_hat
```

Huber loss(delta=1), 투수별 loss를 먼저 평균한 뒤 19명을 동일 가중 평균하는
pitcher-balanced objective, pitcher head L2, 선택적 cumulative ordinal
auxiliary loss를 사용한다. ordered/shuffled, H0/H1/H2,
channels `{4,8}`, bottleneck `{4,8}`, ordinal weight `{0,.25,.5}`,
alpha `{.1,.25,.5}`를 비교했다.

TCN architecture ablation은 receptive field와 같은 L=8에서 사전 고정했고,
ordered H1 기준모델만 L=5를 추가했다. 정상 run은 seeds
`20260722..20260726`을 모두 사용한다.

검증 단계에서 확정한 TCN:

- H1 ordered
- L=8
- internal channels=4
- bottleneck=4
- alpha=0.1
- epochs=250
- learning rate=0.001
- parameter count=248
- ordinal auxiliary weight=0

따라서 `tcn_predictions_2025.parquet`의 probability 열은 스키마상
존재하지만 선택된 residual-only TCN에서는 결측이다. 직접 ordinal 확률은
`ordinal_probabilities_2025.parquet`에 저장된다.

## 6. 평가와 선택 규칙

common sample은 해당 fold의 모든 후보가 예측 가능한 동일 row_id
교집합이다. 주요 모델 비교와 selection은 common sample만 사용한다.
deployable sample은 각 모델이 실제 예측 가능한 전체 target row이고
`n_total_eligible`, `n_predicted`, `coverage`를 함께 보고한다.

TCN은 단일 좋은 seed를 고르지 않는다.

```text
selection score
  = 15개 seed×year balanced accuracy의 평균
  - 0.5 × 그 15개 값의 표준편차
```

비-seeded 모델은 2022/2023/2024 세 값에 같은 식을 적용한다.
최고 점수와 0.003 이내이면 parameter count가 작은 후보를 고르는 simplicity
margin을 적용한다.

보고 지표는 accuracy, balanced accuracy, fixed-3-class macro-F1,
ordinal MAE, Low/Middle/High recall, 양방향 extreme error, 예측 범주 분포,
coverage다.

## 7. 2022~2024 selection 결과

아래는 각 family에서 사전 확정 설정 파일에 저장된 후보의 common-sample
결과다.

| 모델 | Mean BA | Std BA | Selection score | Mean macro-F1 | Ordinal MAE | Seed std | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | **0.561329** | 0.017712 | **0.552473** | 0.565600 | 0.454376 | 0 | 1,900 |
| EWMA4 | 0.558300 | 0.023146 | 0.546727 | 0.563978 | **0.454151** | 0 | 0 |
| Ridge | 0.553851 | 0.023795 | 0.541954 | 0.558657 | 0.461586 | 0 | 323 |
| TCN H1-4 | 0.554810 | 0.026263 | 0.541679 | 0.560529 | 0.458162 | 0.002492 | 248 |
| Ordinal linear | 0.473438 | 0.044491 | 0.451193 | 0.459396 | 0.645967 | 0 | 361 |

선택된 모델의 연도별 balanced accuracy:

| 모델 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| XGBoost | 0.567269 | 0.537285 | 0.579433 |
| EWMA4 | 0.575242 | 0.525573 | 0.574084 |
| Ridge | 0.558154 | 0.522796 | 0.580603 |
| TCN H1-4 seed 평균 | 0.572705 | 0.518121 | 0.573605 |
| Ordinal linear | 0.434106 | 0.450573 | 0.535635 |

TCN ordered H1 L=8의 score는 0.540289이고 같은 기준모델 L=5는
0.539532였다. 선택된 작은 TCN-4는 0.541679였다. H2가 H1보다 안정적으로
우세하지 않았으므로 더 복잡한 pitcher-vector head를 채택하지 않았다.

ensemble raw 결과:

| beta | Mean BA | Std BA | Score | Params |
|---:|---:|---:|---:|---:|
| 0.0 | 0.553851 | 0.023795 | 0.541954 | 323 |
| 0.1 | 0.553387 | 0.022858 | 0.541959 | 571 |
| 0.2 | 0.555065 | 0.023976 | 0.543077 | 571 |
| 0.3 | **0.556742** | 0.025100 | **0.544192** | 571 |

beta=0.3과 beta=0의 score 차이 0.002238은 0.003 simplicity margin보다
작다. 따라서 beta=0을 최종 설정으로 확정했으며, OOF 1,699행에서
beta=0의 연속 예측과 범주가 Ridge와 모두 정확히 같음을 확인했다.

## 8. 사전 확정 설정으로 수행한 2025 최종 결과

### 8.1 Common sample: 동일 497경기

| 모델 | Accuracy | Balanced accuracy | Macro-F1 | Ordinal MAE | Extreme errors |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.501006** | **0.484720** | **0.489101** | 0.559356 | 30 |
| Ridge | 0.498994 | 0.482530 | 0.487167 | **0.557344** | **28** |
| TCN H1-4 | 0.498994 | 0.482675 | 0.485985 | 0.561368 | 30 |
| EWMA4 | 0.490946 | 0.474060 | 0.478178 | 0.569416 | 30 |
| Ordinal linear | 0.474849 | 0.461470 | 0.446586 | 0.661972 | 68 |
| Ridge–TCN beta=0 | 0.498994 | 0.482530 | 0.487167 | **0.557344** | **28** |

common table의 coverage는 정의상 모두 1.0이다.

### 8.2 Deployable sample: 전체 560 target row

| 모델 | Predicted / Eligible | Coverage | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| TCN H1-4 | 560 / 560 | **1.0000** | **0.512500** | **0.498267** | **0.502631** |
| EWMA4 | 560 / 560 | **1.0000** | 0.501786 | 0.487629 | 0.492386 |
| XGBoost | 497 / 560 | 0.8875 | 0.501006 | 0.484720 | 0.489101 |
| Ridge | 497 / 560 | 0.8875 | 0.498994 | 0.482530 | 0.487167 |
| Ridge–TCN beta=0 | 497 / 560 | 0.8875 | 0.498994 | 0.482530 | 0.487167 |
| Ordinal linear | 560 / 560 | **1.0000** | 0.487500 | 0.479667 | 0.463138 |

TCN의 560행 결과와 compact 모델의 497행 결과는 표본이 다르므로
deployable accuracy만으로 모델 우열을 단정하지 않는다. 주요 공정 비교는
앞 절의 common 497행이다.

### 8.3 Ridge와 TCN 진단

- 공통 deployable row: 497
- 범주 agreement: 96.7807%
- disagreement: 16
- residual prediction correlation: 0.325743
- disagreement에서 Ridge accuracy: 43.75%
- disagreement에서 TCN accuracy: 43.75%
- Ridge 오답을 TCN이 수정한 수: 7
- TCN이 새로 만든 extreme error: 2

이 진단 역시 2025 이후 모델을 재선택하는 데 사용하지 않았다.

### 8.4 범주별 재현율과 연속 Stuff+ 오차

동일 common 497행의 범주별 recall:

| 모델 | Low recall | Middle recall | High recall |
|---|---:|---:|---:|
| EWMA4 | 0.560386 | 0.515337 | 0.346457 |
| Ridge | 0.570048 | 0.515337 | 0.362205 |
| XGBoost | **0.574879** | 0.509202 | **0.370079** |
| Ordinal linear | **0.647343** | 0.233129 | 0.503937 |
| TCN H1-4 | 0.555556 | **0.546012** | 0.346457 |
| Ridge–TCN beta=0 | 0.570048 | 0.515337 | 0.362205 |

Ordinal linear는 Low와 High를 더 많이 찾아내는 대신 Middle recall이
23.31%로 낮아 전체 accuracy와 macro-F1이 하락했다. TCN은 Middle
recall이 가장 높지만 High recall이 낮다. 어느 한 모델도 세 범주에서
동시에 우세하지 않다.

범주화 전 연속 Stuff+를 출력하는 모델의 같은 497행 오차:

| 모델 | Stuff+ MAE | Stuff+ RMSE |
|---|---:|---:|
| EWMA4 | 4.3632 | 5.7098 |
| Ridge | 4.3568 | 5.6883 |
| XGBoost | 4.3802 | 5.7050 |
| TCN H1-4 | **4.3415** | **5.6774** |
| Ridge–TCN beta=0 | 4.3568 | 5.6883 |

TCN의 연속 오차가 가장 낮지만 범주 accuracy는 XGBoost보다 낮다.
경계 근처의 작은 오차 변화가 범주 정답 수와 일대일로 대응하지 않기
때문이다. Ordinal linear는 범주 확률을 직접 출력하므로 이 연속 Stuff+
오차 표에는 넣지 않았다.

### 8.5 투수별 common accuracy

아래는 여섯 모델이 모두 예측 가능한 동일 497행만 사용한 투수별
accuracy다. beta=0 ensemble은 Ridge와 같아 중복 열을 생략했다.

| 투수 | N | EWMA4 | Ridge | XGBoost | Ordinal | TCN |
|---|---:|---:|---:|---:|---:|---:|
| José Berríos | 27 | 0.926 | 0.926 | 0.926 | 0.926 | 0.926 |
| Zac Gallen | 30 | 0.733 | 0.733 | 0.733 | 0.800 | 0.733 |
| Luis Castillo | 29 | 0.690 | 0.690 | 0.690 | 0.690 | 0.690 |
| Mitch Keller | 29 | 0.724 | 0.690 | 0.690 | 0.517 | 0.690 |
| Kyle Freeland | 25 | 0.640 | 0.680 | 0.680 | 0.680 | 0.640 |
| Kevin Gausman | 29 | 0.517 | 0.517 | 0.552 | 0.241 | 0.517 |
| Charlie Morton | 22 | 0.455 | 0.500 | 0.500 | 0.591 | 0.500 |
| Zack Wheeler | 21 | 0.476 | 0.476 | 0.476 | 0.333 | 0.476 |
| Brady Singer | 29 | 0.414 | 0.448 | 0.483 | 0.483 | 0.414 |
| Chris Bassitt | 28 | 0.393 | 0.429 | 0.393 | 0.321 | 0.393 |
| Logan Webb | 31 | 0.419 | 0.419 | 0.387 | 0.355 | 0.387 |
| Patrick Corbin | 27 | 0.407 | 0.407 | 0.370 | 0.444 | 0.481 |
| Michael Wacha | 27 | 0.407 | 0.407 | 0.444 | 0.444 | 0.407 |
| Tyler Anderson | 23 | 0.391 | 0.391 | 0.478 | 0.435 | 0.435 |
| Sonny Gray | 28 | 0.286 | 0.357 | 0.286 | 0.429 | 0.321 |
| Jameson Taillon | 17 | 0.353 | 0.353 | 0.353 | 0.294 | 0.353 |
| Dylan Cease | 29 | 0.345 | 0.345 | 0.345 | 0.310 | 0.379 |
| Logan Gilbert | 18 | 0.389 | 0.333 | 0.389 | 0.278 | 0.389 |
| Framber Valdez | 28 | 0.323 | 0.250 | 0.250 | 0.321 | 0.250 |

투수별 accuracy 범위가 매우 넓고 모델별 우위도 투수마다 달라진다.
전체 약 50%라는 값을 모든 투수에게 동일한 품질로 해석해서는 안 된다.
모델별 전체 deployable 표본에서 계산한 모든 투수별 지표 114행은
`final_2025_per_pitcher_metrics.csv`에 있다.

## 9. 생성된 산출물

Selection:

- `experiments/runs/unified_selection/fold_manifest.parquet`
- `experiments/runs/unified_selection/locked_model_config.json`
- `experiments/runs/unified_selection/validation_oof_predictions.parquet`
- `experiments/runs/unified_selection/validation_metrics_by_year.csv`
- `experiments/runs/unified_selection/validation_metrics_summary.csv`
- `experiments/runs/unified_selection/validation_metrics_by_seed.csv`
- `experiments/runs/unified_selection/feature_coverage.csv`
- `experiments/runs/unified_selection/primary_fastball_mapping_by_fold.csv`
- `experiments/runs/unified_selection/xgboost_tpe_trials.csv`

Final:

- `experiments/runs/unified_final_2025/final_2025_predictions.parquet`
- `experiments/runs/unified_final_2025/final_2025_common_sample_metrics.csv`
- `experiments/runs/unified_final_2025/final_2025_deployable_metrics.csv`
- `experiments/runs/unified_final_2025/final_2025_per_pitcher_metrics.csv`
- `experiments/runs/unified_final_2025/ordinal_probabilities_2025.parquet`
- `experiments/runs/unified_final_2025/tcn_predictions_2025.parquet`
- `experiments/runs/unified_final_2025/model_agreement_2025.parquet`
- 모델별 `confusion_matrix_<model>.csv`
- 모델별 `extreme_errors_<model>.csv`
- `feature_coverage.csv`
- `primary_fastball_mapping_by_fold.csv`
- `final_report.json`

`final_2025_predictions.parquet`은 6모델 × 560행 = 3,360행이며
`model,row_id` 중복은 0이다. `tcn_predictions_2025.parquet`은
5 seeds × 560행 = 2,800행이고 요구된 최소 column을 모두 포함한다.

## 10. 테스트와 실제 실행

실행한 검사:

```powershell
python -m pytest -q -rs
python -m compileall -q build_fold_manifest.py select_models.py final_evaluate.py lib tests
git diff --check
python select_models.py --help
python final_evaluate.py --help
```

결과:

- `89 passed, 2 skipped`
- 두 skip은 PyTorch가 설치됐을 때 의도적으로 건너뛰는
  “PyTorch 미설치 오류 경로” 테스트
- compileall 통과
- `git diff --check` 통과
- 두 CLI help 통과
- 실제 raw smoke selection: 99.5초, 성공
- smoke 사전 확정 설정 final 재현: 36.5초, 성공
- 정상 raw selection: 1,191.8초, 성공
- 사전 확정 설정 2025 final: 80.0초, 성공

필수 자동 테스트에는 다음이 포함된다.

- sequence의 최대 game date가 target date보다 과거인지 확인
- 2024 fold의 q33/q67, scaler, imputer, primary type에 2024가 들어가지
  않는지 확인
- validation SI usage가 커도 training FF primary가 바뀌지 않는지 확인
- 40구 공식 선발이 target에서 빠지고 다음 history에는 남는지 확인
- known pitch share 합과 TCN feature 계약 확인
- common row_id 교집합 확인
- q67 tie가 Middle인지 확인
- 여러 fold에서 row-wise q33/q67을 쓰는지 확인
- TCN seed 불안정성이 selection score에서 불리한지 확인
- 사전 확정 설정 파일 없음/fingerprint 불일치/코드 불일치 시 final 실패 확인
- final 코드가 candidate selection을 호출하지 않는지 확인
- ordinal 확률이 음수가 아니고 합이 1인지 확인
- legacy wrapper가 selection/final 책임을 다시 섞지 않는지 확인

## 11. 실행 명령

정상 selection:

```powershell
python select_models.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/mlb_official_pitching_2021_2025.parquet `
  --output-dir experiments/runs/unified_selection
```

사전 확정 설정 final:

```powershell
python final_evaluate.py `
  --statcast-dir data/statcast_mlb_stable_starters_2020 data/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/fangraphs_stuff_mlb_stable_starters_2020.parquet data/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/mlb_official_pitching_2021_2025.parquet `
  --locked-config experiments/runs/unified_selection/locked_model_config.json `
  --output-dir experiments/runs/unified_final_2025
```

## 12. 실패·미실행·한계

- 정상 selection과 final은 모두 성공했다. 결과 숫자를 추정해 채운 부분은
  없다.
- 명시적인 row-level official starter 파일은 제공되지 않았다. raw
  Statcast starter 추론은 경고를 내며, 가능한 환경에서는
  `--starter-status`를 제공하는 편이 안전하다.
- TCN probability 열이 선택된 residual-only TCN에서 결측인 것은 설계상
  정상이다. ordinal 확률 파일은 별도로 생성됐다.
- compact16의 결측 때문에 Ridge/XGBoost의 2025 deployable coverage는
  88.75%다. 이를 숨기지 않고 common/deployable 표를 분리했다.
- 현재 unified final은 Ridge 계수표와 학습된 model weight를 별도
  직렬화하지 않는다. 원자료에서 사전 확정 설정을 그대로 다시
  학습·평가하는 재현 파이프라인이며, 즉시 서빙하는 모델 패키지는 아니다.
- `final_2025_predictions.parquet`의 `pitcher_name`은 tabular 모델
  행에는 있으나 EWMA·ordinal·TCN 행에는 결측이다. `pitcher` ID와
  row_id, 성능 계산에는 영향이 없으며 이름이 필요한 교차 모델 표에서는
  투수 ID 기준으로 tabular 이름을 결합해야 한다.
- `data/*`와 `experiments/runs/*`는 크기와 재생성 가능성을 이유로
  `.gitignore` 대상이다. Git에는 코드와 보고서가 들어가고, 이 보고서가
  인용한 실행 산출물은 로컬에 보존된다.
- `FINAL_MODEL_REPORT.html`은 과거 legacy 보고서 스냅샷이며 이번 통합
  Markdown 보고서로 재생성하지 않았다. 해당 PDF와 현재 결과의 비교는
  `LEGACY_VS_CURRENT_COMPARISON_REPORT.md`에 별도로 기록했다.
- 2025는 단 한 시즌의 최종 진단이다. 이 결과로 하이퍼파라미터나 모델
  역할을 다시 선택하지 않았다.
