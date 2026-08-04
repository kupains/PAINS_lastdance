# MLB 선발투수 Stuff+ 범주 예측 최종 종합 보고서

- 작성일: 2026-07-24
- 주 예측 모델: XGBoost (내부 실행명: `xgboost_tpe_span4_compact_locked`)
- 해석 보조 모델: Ridge (내부 실행명: `ridge_span4_compact_alpha100_scale0.5`)
- 개발 데이터: 2020~2024년
- 최종 성능평가 데이터: 2025년
- 분석 단위: 투수별 50구 이상 공식 선발 등판
- 최종 평가 표본: 19명, 497등판

## 1. 요약

2020~2024년 자료만으로 모델과 설정을 모두 정한 뒤, 모델 선택에 사용하지 않은
2025년 자료로 최종 성능을 평가했다. 최종 모델은 비선형 관계를 학습하는 XGBoost와
결과를 계수로 설명하기 쉬운 Ridge 두 가지다.

| 역할 | 모델 | 2025 정확도 | 범주별 평균 정확도 | 범주별 평균 F1 | 범주 거리 오차 |
|---|---|---:|---:|---:|---:|
| 주 예측 | XGBoost | 51.71% | 49.31% | 0.496 | 0.547 |
| 해석 보조 | Ridge | **52.92%** | **51.05%** | **0.514** | **0.531** |

두 모델은 497건 중 455건에서 같은 범주를 예측해 일치율 91.55%를 기록했다. 일치한
등판의 정확도는 53.41%였다. 불일치 42건에서는 XGBoost가 33.33%, Ridge가 47.62%를
맞혔다. 이 결과는 설명용 진단이며 2025를 보고 모델 설정이나 역할을 변경하지 않았다.

실무적으로는 XGBoost의 비선형 예측과 Ridge의 해석을 함께 제공하되, 두 모델의 예측 범주가
다르면 불확실성이 높은 등판으로 표시하는 방식이 적절하다. 현재 성능은 다음 등판의 변화를
정확히 맞히는 수준이라기보다, 최근 4경기에 더 큰 비중을 둔 Stuff+ 평균을 기본값으로 삼고
물리·워크로드 피처로 조금 보정하는 수준이다.

### 결과표 읽는 법

- **정확도**: 전체 497등판 중 예측 범주를 정확히 맞힌 비율이다.
- **범주별 평균 정확도**: Low·Middle·High 각각의 정답률을 동일한 비중으로 평균한 값이다.
  표본이 많은 범주에만 잘 맞혀도 높아지는 문제를 줄인다.
- **범주별 평균 F1**: 각 범주의 정밀도와 재현율을 함께 반영한 뒤 세 범주를 평균한 값이다.
- **범주 거리 오차**: 정확하면 0, 한 단계 차이면 1, Low와 High처럼 두 단계 차이면 2다.
  **낮을수록 좋다.**
- **최근 4경기 가중평균(EWMA4)**: 최근 등판일수록 더 큰 비중을 주어 계산한 Stuff+ 평균이다.

## 2. 모델 선택과 최종 평가 방식

이번 실행은 다음 순서를 강제한다.

1. 모델 선택 함수에는 2024년 이하 데이터만 전달한다.
2. 2020~2021 학습 → 2022 검증을 수행한다.
3. 2020~2022 학습 → 2023 검증을 수행한다.
4. 2020~2023 학습 → 2024 검증을 수행한다.
5. 연도별 범주 평균 정확도의 평균에서 변동성의 절반을 뺀 점수로 XGBoost를 확정한다.
6. XGBoost가 Ridge보다 범주별 평균 정확도에서 0.5%p 이상 높아야 주 모델 조건을 통과한다.
7. 두 모델을 2020~2024년으로 재학습한 뒤 2025년에 적용한다.

최종 XGBoost는 Ridge보다 연도별 검증의 범주 평균 정확도가 0.517%p 높아 기준을 간신히
통과했다. 두 모델의 세부 설정은 모두 2025년 결과를 보기 전에 정했다.

## 3. 데이터

### 3.1 출처

| 데이터 | 사용 내용 |
|---|---|
| Statcast | 투구 수, 구속, 회전수, 릴리스, 무브먼트, 구종 구성 |
| FanGraphs 경기 로그 | 공식 선발 여부와 선발 Stuff+ |
| MLB 공식 투구 기록 | 선수 식별자와 이름 |

2020년은 단축 시즌이므로 선수 자격 판정에는 사용하지 않고 추가 학습 이력으로만 사용했다.

### 3.2 분석 대상

2021~2025년의 모든 시즌에서 `공식 선발 + Statcast 50구 이상` 등판을 20회 이상 확보한
19명을 대상으로 했다. 2025년 등판 수는 평가 선수를 정하는 데 사용했지만, 2025년
Stuff+ 결과는 모델 종류나 세부 설정을 선택하는 데 사용하지 않았다.

### 3.3 시즌별 표본

| 시즌 | 50구 이상 선발 | 결측값 없는 등판 | 결측 제외 |
|---:|---:|---:|---:|
| 2020 | 176 | 118 | 58 |
| 2021 | 512 | 436 | 76 |
| 2022 | 546 | 482 | 64 |
| 2023 | 583 | 520 | 63 |
| 2024 | 570 | 502 | 68 |
| 2025 | 560 | 497 | 63 |

입력값 또는 예측할 범주에 결측이 있는 행은 학습과 평가에서 제외했다. 두 최종 모델과
비교용 단순 예측은
동일한 2025년 497등판에서 비교했다.

## 4. 예측할 범주

각 투수의 학습 구간 Stuff+에서 33.3%와 66.7% 분위수를 계산했다.

- Low: Stuff+ ≤ 학습 구간 33.3% 분위수
- Middle: 두 분위수 사이
- High: Stuff+ > 학습 구간 66.7% 분위수

각 연도를 검증할 때는 그보다 이전 시즌의 학습 자료만으로 경계값을 만든다. 최종 2025 평가는
2020~2024년 Stuff+만으로 선수별 경계값을 만든다.

## 5. 모델에 넣은 정보

### 5.1 공통 후보 28개

워크로드:

- `prev_start_pitch_count`
- `rest_days`
- `workload_density_3starts`

Stuff+ 이력:

- `prior_stuff_plus`
- `stuff_plus_mean_last5`
- `stuff_plus_slope_last5`

투구 물리 정보:

- `release_speed_ma5`, `release_speed_slope5`
- `release_spin_rate_ma5`, `release_spin_rate_slope5`
- `release_extension_ma5`, `release_extension_slope5`
- `release_pos_x_ma5`, `release_pos_x_slope5`
- `release_pos_z_ma5`, `release_pos_z_slope5`
- `arm_angle_ma5`, `arm_angle_slope5`
- `pfx_x_ma5`, `pfx_x_slope5`
- `pfx_z_ma5`, `pfx_z_slope5`
- `spin_axis_sin_ma5`, `spin_axis_cos_ma5`

구종 구성:

- `breaking_share_ma5`, `breaking_share_slope5`
- `offspeed_share_ma5`, `offspeed_share_slope5`

구종은 패스트볼·브레이킹볼·오프스피드로 묶었다. 합계 제약으로 인한 중복을 피하기 위해
패스트볼 비중은 직접 입력하지 않았다.

### 5.2 최종 모델에 사용한 16개 입력 정보

두 최종 모델은 후보 28개 중 동일하게 선별한 16개 피처를 사용한다.

- `prior_minus_ewma4`
- `mean5_minus_ewma4`
- `stuff_plus_slope_last5`
- `prev_start_pitch_count`
- `rest_days`
- `workload_density_3starts`
- `release_speed_slope5`
- `release_spin_rate_slope5`
- `release_extension_slope5`
- `release_pos_x_slope5`
- `release_pos_z_slope5`
- `arm_angle_slope5`
- `pfx_x_slope5`
- `pfx_z_slope5`
- `breaking_share_slope5`
- `offspeed_share_slope5`

## 6. 모델 구조

두 모델 모두 최근 4경기 가중평균(EWMA4)을 다음 경기의 기본 예상값으로 둔다. 그다음
물리·워크로드 피처로 실제 Stuff+가 이 기본값에서 얼마나 벗어날지를 예측한다.

```text
기본값과의 차이 = 현재 Stuff+ - 직전까지의 최근 4경기 가중평균
최종 예상 Stuff+ = 최근 4경기 가중평균 + 보정 반영률 × 예상 차이
```

### 6.1 XGBoost

- 역할: 비선형 주 예측
- 입력 피처: 최종 선별 16개
- 기본 예상값: 최근 4경기 가중평균
- trees: 100
- max depth: 2
- learning rate: 0.04221
- min child weight: 15.043
- L2(`reg_lambda`): 16.823
- L1(`reg_alpha`): 0.706
- gamma: 0
- subsample: 0.9997
- column sample: 0.5856
- 모델 보정 반영률: 57.5%
- tree method: histogram, max bins 64

### 6.2 Ridge

- 역할: 해석 가능한 보조 예측
- 입력 피처: 최종 선별 16개
- 기본 예상값: 최근 4경기 가중평균
- alpha: 100
- 모델 보정 반영률: 50%
- 입력 피처: 투수별 학습 데이터에서 표준화

Ridge는 과적합을 막는 제약을 강하게 적용하고, 모델이 계산한 보정값도 50%만 반영한다.
따라서 최근 4경기 가중평균에서 크게 벗어나지 않는 보수적인 예측을 만든다.

## 7. 연도별 검증 결과와 모델 선택

XGBoost의 세부 설정은 자동 탐색 방법(TPE)으로 다음과 같이 좁혔다.

1. 자동으로 만든 설정 32개를 2023년에 평가하고 상위 10개 유지
2. 상위 10개를 2023~2024년에 평가하고 상위 3개 유지
3. 상위 3개를 2022~2024년에 평가
4. 최고 점수와 0.3%p 이내로 비슷하면 과적합 위험이 낮은 더 단순한 설정 선택

선택 점수는 `범주별 평균 정확도의 연도 평균 - 0.5 × 연도별 변동성`이다. 여러 해에 걸쳐
성능이 높고 안정적인 설정을 고르기 위한 기준이다.

| 모델 | 2022 | 2023 | 2024 | 평균 | 표준편차 | 선택 점수 |
|---|---:|---:|---:|---:|---:|---:|
| 최적화 XGBoost | 54.27% | 53.57% | 55.83% | **54.56%** | **0.94%p** | **54.09%** |
| Ridge | 53.39% | 52.98% | 55.76% | 54.04% | 1.23%p | 53.43% |
| EWMA4 | 54.65% | 51.99% | 55.67% | 54.10% | 1.55%p | 53.33% |

XGBoost의 Ridge 대비 범주별 평균 정확도 우위는 0.517%p로 사전에 정한 0.5%p 조건을
통과했다. 동일한 난수 기준으로 전체 탐색을 두 번 실행했을 때 최종 설정이 완전히 같았고,
각 실행 시간은 약 29초였다.

## 8. 2025 최종 성능평가

| 모델 | 평가 등판 | 정확도 | 범주별 평균 정확도 | 범주별 평균 F1 | 범주 거리 오차 |
|---|---:|---:|---:|---:|---:|
| **Ridge 해석 보조** | 497 | **52.92%** | **51.05%** | **0.514** | **0.531** |
| EWMA4 단독 | 497 | 52.31% | 50.65% | 0.510 | 0.545 |
| XGBoost 주 예측 | 497 | 51.71% | 49.31% | 0.496 | 0.547 |
| 직전 Stuff+ | 497 | 49.90% | 47.94% | 0.480 | 0.610 |
| 학습기간에서 가장 많았던 범주 | 497 | 42.86% | 33.33% | 0.200 | 0.819 |

Ridge가 XGBoost보다 6건 더 맞혔고 EWMA4보다는 3건 더 맞혔다. 최적화 XGBoost는 기존
고정 XGBoost보다 4건을 추가로 맞혔다. 2025년 결과는 두 모델의 설명과 비교에만 사용했으며,
이 결과를 근거로 모델 설정을 다시 바꾸지 않았다.

### 8.1 XGBoost 혼동행렬

| 실제＼예측 | Low | Middle | High |
|---|---:|---:|---:|
| Low | **129** | 70 | 14 |
| Middle | 35 | **87** | 39 |
| High | 18 | 64 | **41** |

### 8.2 Ridge 혼동행렬

| 실제＼예측 | Low | Middle | High |
|---|---:|---:|---:|
| Low | **128** | 70 | 15 |
| Middle | 36 | **87** | 38 |
| High | 15 | 60 | **48** |

Ridge는 XGBoost보다 High를 7건 더 맞혔고 Low는 1건, Middle은 같았다.

## 9. 두 모델의 일치와 활용

| 구분 | 등판 수 | 비율 또는 정확도 |
|---|---:|---:|
| 같은 범주 예측 | 455 | 91.55% |
| 다른 범주 예측 | 42 | 8.45% |
| 일치 구간 정확도 | 455 | 53.41% |
| 불일치 구간 XGBoost 정확도 | 42 | 33.33% |
| 불일치 구간 Ridge 정확도 | 42 | 47.62% |

권장 출력 방식은 다음과 같다.

- 두 모델의 범주가 같으면 `agreement=True`로 함께 제시한다.
- 다르면 예측 불확실성이 높은 등판으로 표시한다.
- XGBoost 결과에는 Ridge 결과와 주요 계수 방향을 함께 제공한다.
- 현재 데이터만으로 두 모델의 평균이나 투표 규칙을 새로 조정하지 않는다.

## 10. 선수별 2025 결과

| 투수 | N | XGBoost | Ridge | 두 모델 일치율 |
|---|---:|---:|---:|---:|
| José Berríos | 27 | 92.6% | 92.6% | 100.0% |
| Mitch Keller | 29 | 82.8% | 82.8% | 100.0% |
| Zac Gallen | 30 | 76.7% | 73.3% | 96.7% |
| Luis Castillo | 29 | 75.9% | 69.0% | 93.1% |
| Kyle Freeland | 25 | 60.0% | 64.0% | 88.0% |
| Kevin Gausman | 29 | 55.2% | 58.6% | 89.7% |
| Zack Wheeler | 21 | 57.1% | 57.1% | 100.0% |
| Brady Singer | 29 | 55.2% | 51.7% | 93.1% |
| Charlie Morton | 22 | 45.5% | 50.0% | 90.9% |
| Michael Wacha | 27 | 44.4% | 48.1% | 96.3% |
| Tyler Anderson | 23 | 56.5% | 47.8% | 91.3% |
| Jameson Taillon | 17 | 41.2% | 47.1% | 88.2% |
| Logan Gilbert | 18 | 38.9% | 44.4% | 94.4% |
| Sonny Gray | 28 | 32.1% | 42.9% | 85.7% |
| Logan Webb | 31 | 35.5% | 38.7% | 90.3% |
| Chris Bassitt | 28 | 42.9% | 35.7% | 92.9% |
| Dylan Cease | 29 | 24.1% | 34.5% | 82.8% |
| Patrick Corbin | 27 | 22.2% | 33.3% | 88.9% |
| Framber Valdez | 28 | 35.7% | 28.6% | 78.6% |

선수별 편차가 매우 크다. 상위 선수에서는 한 시즌의 범주가 특정 상태에 오래 머문 반면,
하위 선수에서는 2025 분포 변화와 경기별 변동을 두 모델 모두 충분히 설명하지 못했다.
따라서 전체 정확도를 모든 투수에게 동일하게 적용되는 품질로 해석하면 안 된다.

## 11. Ridge가 중요하게 본 입력 정보

아래 값은 각 입력 정보가 기본 예상값을 어느 방향으로 얼마나 보정했는지를 나타낸다.
선수마다 단위가 다른 문제를 줄이기 위해 각 투수의 학습 데이터 안에서 입력값의 척도를
맞췄으며, 표에는 19명 계수의 중앙값을 제시했다.

| 입력 정보 | 계수 중앙값 | 영향 크기 중앙값 | 양수 선수 비율 | 해석 |
|---|---:|---:|---:|---|
| `stuff_plus_slope_last5` | -0.189 | 0.189 | 0.0% | 최근 상승 뒤 낮아지고 하락 뒤 높아지는 방향으로 보정 |
| `prior_minus_ewma4` | -0.138 | 0.138 | 15.8% | 직전 경기의 EWMA 이탈을 반대 방향으로 조정 |
| `release_spin_rate_slope5` | -0.001 | 0.128 | 47.4% | 영향 크기는 있으나 선수별 방향이 다름 |
| `mean5_minus_ewma4` | +0.119 | 0.119 | 94.7% | 최근 5경기 평균이 EWMA보다 높으면 양의 보정 |
| `arm_angle_slope5` | +0.039 | 0.113 | 57.9% | 선수별 이질성이 큰 보조 신호 |
| `prev_start_pitch_count` | -0.041 | 0.107 | 31.6% | 직전 투구 수 증가가 대체로 약한 음의 보정 |
| `release_speed_slope5` | -0.016 | 0.107 | 36.8% | 구속 변화의 방향은 선수별로 다름 |

가장 일관된 신호는 최근 크게 오르거나 내린 Stuff+가 다시 평소 수준으로 돌아오는 경향이다.
`stuff_plus_slope_last5`는 19명 모두 음수였고,
`mean5_minus_ewma4`는 19명 중 18명에서 양수였다. 반면 회전수·암 슬롯·구속 변화는 절댓값이
커도 부호가 엇갈리므로 전 리그에 동일한 인과 방향이 있다고 해석하면 안 된다.

## 12. 한계

1. 최종 성능평가가 2025년 한 시즌에만 이루어졌다.
2. 투수별 모델이라 각 모델의 학습 표본이 작다.
3. 입력값이나 정답에 결측이 있는 2025년 63등판을 제외했다.
4. 2021~2025년 매년 20회 이상 선발한 투수만 포함했으므로 부상·보직 변경이 있었던 투수에게
   같은 성능을 기대하기 어렵다.
5. 2025년 선발 등판 수까지 확인해 평가 선수를 정했기 때문에 표본 선택 편향이 있을 수 있다.
6. 최종 코드는 2025년 결과를 모델 선택에 사용하지 않지만, 연구자가 초기 탐색 과정에서
   2025년 결과를 확인했던 사실은 되돌릴 수 없다.
7. Ridge를 보조 모델로 함께 제시하기로 한 결정은 첫 XGBoost 최종 평가 이후 이루어졌다.
   Ridge 설정 자체는 2023·2024년 검증만으로 정했지만, Ridge의 2025년 성능은 보조 결과로
   해석하는 것이 안전하다.

## 13. 재현 방법

```powershell
.\.venv\Scripts\python.exe search_xgboost_fast.py `
  --statcast-dir data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2020 data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2020.parquet data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --output-dir experiments/runs/xgboost_fast_search_2020_2025

.\.venv\Scripts\python.exe stuff_mlb_temporal_final.py `
  --statcast-dir data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2020 data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2020.parquet data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/reproducible_mlb_stuff/mlb_official_pitching_2021_2025.parquet `
  --output-dir experiments/runs/mlb_stuff_temporal_selected_2020_2025
```

| 파일 | 내용 |
|---|---|
| `locked_xgboost_config.json` | 결정론적으로 잠근 XGBoost 파라미터와 검증·진단 성능 |
| `stage1_32_candidates_2023.csv` | 1단계 TPE 후보 |
| `stage2_top10_2023_2024.csv` | 2단계 생존 후보 |
| `stage3_top3_2022_2024.csv` | 최종 3개 후보 |
| `locked_xgboost_2025_predictions.parquet` | 최종 XGBoost의 2025 예측 |
| `tuned_xgboost_vs_ridge_2025.parquet` | 최적화 XGBoost와 Ridge 경기별 비교 |
| `validation_candidates.csv` | 2025 제외 후보 222개의 검증 성능과 순위 |
| `pooled_metrics.csv` | 두 모델과 기준선의 2025 전체 성능 |
| `per_pitcher_metrics.csv` | 선수별 모델 성능 |
| `dual_model_comparison.parquet` | 경기별 두 모델 예측과 일치 여부 |
| `ridge_coefficients.csv` | 19명 × 16피처 표준화 계수 |
| `ridge_feature_summary.csv` | 계수 중앙값·사분위수·부호 일관성 |
| `confusion_matrix_xgboost.csv` | XGBoost 혼동행렬 |
| `confusion_matrix_ridge.csv` | Ridge 혼동행렬 |
| `predictions.parquet` | 경기별 모든 예측 |
| `report.json` | 선택 규칙, 이중 모델 설정, 성능과 일치율 |

## 14. 결론

XGBoost와 Ridge는 2025 이전 검증만으로 설정을 확정했다. 최적화 XGBoost는 Ridge보다
검증의 범주별 평균 정확도가 0.517%p 높아 주 모델 조건을 통과했다. 2025년에는 Ridge가
정확도 52.92%로 XGBoost의 51.71%보다 높았고, 두 모델은 91.55%의 등판에서 같은 범주를
냈다. 따라서 XGBoost의 비선형 예측과 Ridge의 안정적·해석 가능한
보조 예측을 함께 제공한다. 다음 시즌에는 설정을 다시 손대지 않고 실제 새 등판에 적용해
성능이 유지되는지 확인해야 한다.

최종적으로 이 모델의 실질적인 기반은 최근 Stuff+ 흐름을 반영한 4경기 가중평균이다.
XGBoost와 Ridge는 이 기본 예상값을 조금 조정하며, 2025년에서 가장 좋은 Ridge도
4경기 가중평균만 사용했을 때보다 3경기를 더 맞힌 정도다. 그러므로 복잡한 모델 자체보다
모델 선택 자료와 최종 평가 자료를 분리하고, 과적합을 강하게 억제하며, 두 모델의 일치 여부와
선수별 성능 편차를 함께 보여주는 것이 더 중요하다.
