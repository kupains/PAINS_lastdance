# 3범주 분류 베이스라인 명세

> 이 문서는 **단일 등판** 베이스라인 명세다. 예측 성능 최적 결과는 단일 등판
> 노이즈를 제거한 다중등판 rolling 타깃이며, `docs/rolling_target_results.md`와
> `run_rolling.py`를 참조. (단일등판 baseline lift 1.207 → rolling k=5 1.576 →
> k=10 1.828)

## 목적

등판 직전 시점에서 아래 질문에 답한다.

```text
오늘 이 투수를 내도 되는가?
본인 평소 실력 대비 하/중/상 중 어느 범주의 등판이 기대되는가?
```

현재 베이스라인은 회귀가 아니라 3범주 분류다.

## 예측 단위

```text
pitcher-game relief outing
```

즉 한 투수의 한 경기 불펜 등판을 하나의 row로 본다.

## 표본 기준

현재 MLB 전체팀 베이스라인 필터:

```text
기간: 2021-2025
투수별 최소 30 IP
투수별 최소 100 BF
```

이 기준은 투수별 prior, baseline, workload z-score를 계산하기 위한 최소 안정성 기준이다.

## 라벨

먼저 등판별 상대 성과를 계산한다.

```text
residual = baseline_skill - shrunk_xwOBA
```

xwOBA는 낮을수록 좋기 때문에:

```text
residual > 0 : 본인 baseline보다 좋은 등판
residual < 0 : 본인 baseline보다 나쁜 등판
```

그다음 train 구간에서만 `residual`의 1/3, 2/3 분위수를 계산해 3개 클래스를 만든다.

```text
0 = 하 / risk   : residual <= train 1/3 quantile
1 = 중 / normal : train 1/3 quantile < residual < train 2/3 quantile
2 = 상 / good   : residual >= train 2/3 quantile
```

컷포인트는 train 구간에서만 계산하고 test에는 그대로 적용한다. 미래 정보를 쓰지 않기 위해서다.

## shrunk_xwOBA

불펜 등판은 보통 BF가 작아서 raw `outing_xwOBA`가 너무 흔들린다. 그래서 현재 등판 xwOBA를 투수 개인 prior 쪽으로 shrink한다.

```text
shrunk_xwOBA =
(BF * outing_xwOBA + eb_k * personal_prior_xwOBA) / (BF + eb_k)
```

기본값:

```text
eb_k = 10
```

BF가 작을수록 개인 prior를 더 믿고, BF가 클수록 해당 등판의 raw xwOBA를 더 반영한다.

## 입력 피처

현재 feature set:

```text
personalized_workload_max
```

포함:

- workload 원값: `acute_workload_7d`, `chronic_workload_28d`, `ACWR`, `rest_days`, `back_to_back`
- standard abuse rolling feature
- 투수 개인 기준 workload/abuse z-score: `*_pitcher_z`
- Statcast tracking rolling feature: `*_ma5`, `*_slope5`, `*_z`
- pitch mix rolling feature: `pitch_mix_*_ma5`, `pitch_mix_*_slope5`, `pitch_mix_*_z`
- `pitcher_prior_outing_count`

제외:

- `role`: 현재 데이터에서 의미 있는 역할 구분으로 작동하지 않아 baseline에서 제외
- `baseline_skill`, `shrunk_xwOBA`, `outing_xwOBA`, `residual`, `target_y`: 결과/라벨 계열이라 입력 피처에서 제외

## 모델

현재 유일한 베이스라인 모델:

```text
classification_residual_tertile_xgboost
```

학습 방식:

```text
objective: multi:softprob
num_class: 3
sample_weight: BF
class_weight: balanced
```

BF 가중치는 더 많은 타자를 상대한 등판을 더 신뢰하기 위한 장치이고, 클래스 밸런스 가중치는 모델이 `중/normal`에만 몰리는 것을 막기 위한 장치다.

## 실행 명령

```bash
python compare.py ^
  --input data/outings_mlb_bullpen_2021_2025.parquet ^
  --run-id mlb_bullpen_2021_2025_min30ip100bf_3class_baseline ^
  --output-dir experiments/runs/mlb_bullpen_2021_2025_min30ip100bf_3class_baseline ^
  --min-pitcher-ip 30 ^
  --min-pitcher-bf 100
```

`--models`를 생략하면 이 베이스라인 모델만 실행한다.

## 평가 지표

```text
accuracy
balanced_accuracy
macro_f1
risk_precision
risk_recall
risk_f1
top20_risk_rate
top20_risk_lift
```

의사결정에서는 `accuracy`보다 아래 지표를 더 중요하게 본다.

```text
risk_precision : 모델이 하/risk라고 찍은 등판 중 실제 하/risk 비율
risk_recall    : 실제 하/risk 등판 중 모델이 잡아낸 비율
top20_risk_lift: risk 확률 상위 20%가 전체 평균 대비 얼마나 위험 등판을 농축하는지
```

## 현재 MLB 베이스라인 결과

실행 결과:

```text
experiments/runs/mlb_bullpen_2021_2025_min30ip100bf_3class_baseline
```

지표:

```text
accuracy          0.371917
balanced_accuracy 0.365988
macro_f1          0.353410
risk_precision    0.401074
risk_recall       0.184059
top20_risk_rate   0.389662
top20_risk_lift   1.207408
```

해석:

```text
무작위 3분류 기준선인 balanced_accuracy 약 0.333보다는 낫다.
하지만 강한 판별기라고 보기는 어렵다.
현재는 등판 결과를 정확히 맞히는 모델이라기보다 risk 후보를 약하게 랭킹하는 베이스라인이다.
```
