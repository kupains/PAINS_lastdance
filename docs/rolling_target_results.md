# 다중등판 rolling 타깃 결과

## 배경

단일 등판 3범주 베이스라인은 top20_risk_lift 1.207로 약했다. 두 진단이 원인을
모델이 아니라 **타깃**으로 지목했다:

- `docs/response_heterogeneity_validation.md`: 투수별 워크로드 반응은 시간
  전이가 안 됨(상관 ≈ 0). 아키텍처(클러스터/트랜스포머/GNN) 교체로는 못 얻음.
- `experiment_label_debias.py`: 단일 등판 residual은 BF≈4에서 ~99% 표본 노이즈.
  정직한 within-pitcher 신호의 예측 상한이 lift ~1.11(랜덤 근처).

해법: 등판 노이즈를 **forward rolling 평균**으로 제거해 예측 가능한 신호를 복원.

## 타깃 정의

투수별, 날짜 정렬:

```text
rolling_fwd{k}_residual          = mean(residual[t .. t+k-1])   # 오늘 포함 향후 k등판
pitcher_offset_prior             = reliability * expanding_mean(residual[<t])   # prior-only, shrunk
rolling_fwd{k}_residual_centered = rolling_fwd{k}_residual - pitcher_offset_prior
```

- forward window라 결정 시점 t의 예측(forecast)이 됨. trailing은 이미 알므로 타깃 아님.
- 같은 시즌 내 완전한 k등판 window만 사용(오프시즌 gap 배제).
- offset은 `n/(n+10)` shrinkage. 단일 residual sd(~0.24)가 커서 소표본 prior mean을
  그대로 빼면 노이즈를 주입 → shrink로 방지. offset sd 0.025(투수별 평균 residual과 일치).
- **Leakage 처리**: `rolling_fwd{k}_window_end_date`(t+k-1 날짜)를 만들어, split 후
  window 끝이 embargo 경계를 넘는 train 행을 제거(`slice_train_test`). offset은 t 이전만
  쓰므로 forward window와 겹치지 않음. 코드: `lib/rolling_target.py`.

구현: `lib/rolling_target.py`, 실험: `experiment_rolling_target.py`,
프로덕션: `run_rolling.py`, 모델: `models/model_classification_rolling_residual_tertile_xgboost.py`.

## k 스윕 (동일 rolling-origin split, top20_risk_lift)

| k | raw base | raw +prior | debiased base | debiased +prior |
|---|---|---|---|---|
| 1 (단일등판) | 1.207 | 1.205 | 1.071 | 1.195 |
| 3 | 1.356 | 1.458 | 1.149 | 1.430 |
| 5 | 1.444 | 1.564 | 1.256 | **1.557** |
| 10 | 1.745 | **1.824** | 1.280 | 1.816 |

- base = `personalized_workload_max`, +prior = `decision_pregame_shrunk_xwoba`
  (base + `personal_prior_xwOBA`, `normal_condition_count_prior`).
- 단일등판(k=1) 값은 `experiment_label_debias.py`의 A(1.207)·D(1.109)를 동일 하네스로 재현.

## 성능 출처 분해 (정직성)

denoise는 노이즈만 줄인다. 늘어난 lift가 어디서 오는지 분해하면:

```text
debiased_base (offset 제거 + identity 피처 차단):
  k=1..10 = 1.071 -> 1.149 -> 1.256 -> 1.280   (거의 평평)
```

→ 순수 within-pitcher form/workload 신호는 denoise해도 약함. 반면 prior-skill
피처를 넣으면 k=10에서 1.28 -> 1.82로 급등. **랜덤(1.0) 위 초과분의 약 2/3가
pitcher talent/identity, 약 1/3이 time-local form 신호.** 워크로드 자체 기여는 미미.

원리적 주의: k가 커질수록 타깃은 "career 평균 residual = 정적 talent"로 수렴한다.
큰 k의 높은 lift는 "오늘 기용 결정"이 아니라 "투수 기량 랭킹"에 가까워진다.

## 채택 구성

```text
기본 (shipped):  k=5, centered, decision_pregame_shrunk_xwoba
  -> top20_risk_lift 1.576, balanced_accuracy 0.411, risk_precision 0.482
  근거: baseline 1.207 대비 +31%, ~2주 결정 지평으로 의사결정 의미 유지,
        centered 타깃으로 within-pitcher 프레이밍 정직.

최고 성능:       k=10, centered/raw, decision_pregame_shrunk_xwoba
  -> top20_risk_lift ~1.82
  주의: 사실상 talent 랭킹에 근접. 순수 지표만 필요할 때.
```

재현:

```bash
python run_rolling.py --k 5                  # 기본
python run_rolling.py --k 10 --variant raw   # 최고 lift
```

## 정직성/한계

- 이 모델은 "향후 k등판 구간이 본인 baseline 대비 좋/나쁠까"를 예측하며, 신호의
  대부분은 talent/form이고 workload가 아니다. 유효한 예측 모델이나 "워크로드 리스크
  탐지기"라기보다 "폼/기량 구간 예측기"다.
- forward window는 결정을 "오늘 1경기"에서 "향후 k등판"으로 흐린다(k↑ 신호↑·즉시성↓).
- 관측 데이터 선택편향(감독이 지친 투수 회피)은 여전. 인과 해석 금지.
- 표본: MLB 2021-2025, 30 IP + 100 BF 필터. test = 2025 후반기 rolling-origin split.
