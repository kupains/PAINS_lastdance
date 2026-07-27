# 워크로드 반응 개인화 전제 검증

## 목적

반응 프로파일 개인화(클러스터링, 트랜스포머, GNN 등 무엇이든)가 성립하려면 아래 두 전제가 필요하다.

```text
(a) 투수별 워크로드 -> 성과 반응이 실제로 이질적이다
(b) 그 이질성이 시간적으로 안정적이다 (과거로 추정해 미래에 적용 가능)
```

이 문서는 그 전제를 직접 측정한 결과다.

실행:

```bash
python validate_heterogeneity.py
```

산출물: `experiments/runs/response_heterogeneity_validation/`

## 설계

- 표본: MLB 2021-2025, 30 IP + 100 BF 필터, complete-case 77,741 등판 / 798 투수
- 예측변수 5개: `ACWR`, `rest_days`(14일 캡), `back_to_back`, `standard_abuse_sum_7d`, `release_speed_z`
- 투수-반쪽 단위로 within-pitcher 표준화 후, pooled 회귀(공용 반응 `beta_global`) + 투수별 ridge 편차(`delta`, lambda 10/25/50)
- 두 가지 반쪽 분할:
  - `odd_even`: 홀짝 등판 교차 — 추정 신뢰도의 상한 (시간 이동 통제)
  - `chrono`: 커리어 전반부 vs 후반부 — 실전 배치와 같은 시간 전이 안정성
- 판정 지표:
  - `delta`의 split-half 상관 (계수 수준 안정성)
  - prediction transfer: 반쪽 A의 개인 계수가 반쪽 B를 공용 계수보다 잘 예측하는가 (의사결정 기준 지표)

## 사전 점검

- complete-case 93.9% (ACWR 초기 결측 4,253행, release_speed_z 1,783행 탈락)
- 투수 내 변동 충분: ACWR std 중앙값 0.98, rest_days(캡) 2.1일 — 기울기 식별 가능
- 공선성 낮음: 최대 |r| = 0.45 (rest_days vs back_to_back), 나머지 |r| < 0.36 — 계수 회전으로 상관이 붕괴할 수준 아님
- 양쪽 반기 25등판 이상 투수 474명, 40등판 이상 358명 — 상관 추정에 충분

## 결과

### 1. 공용(전 투수 평균) 워크로드 반응 자체가 사실상 0이다

```text
pooled within-pitcher R^2 (odd_even):  +0.0004
pooled within-pitcher R^2 (chrono):    -0.0005  (다른 시대로 전이 실패)
계수 크기: 1SD당 |beta| <= 0.0017 xwOBA  (tertile 컷 ±0.026의 1/15 수준)
```

워크로드 피처의 선형 within-pitcher 신호는 baseline 대비 성과를 거의 설명하지 못한다.

### 2. 개인별 반응 편차는 시간 전이가 되지 않는다

계수 수준 split-half 상관 (lambda=25, min 25등판/반쪽):

```text
                          odd_even      chrono
ACWR                       0.49          -0.02
release_speed_z            0.50          -0.01
standard_abuse_sum_7d      0.27          -0.01
rest_days_capped           0.04           0.04
back_to_back              -0.07          -0.03
```

chrono에서는 다섯 계수 모두 CI가 0을 포함한다. odd_even의 높은 상관은 인접 등판의
rolling window 중첩(ACWR, abuse_sum이 창을 공유)으로 부풀려진 상한일 뿐이며,
그 낙관적 설계에서도 prediction transfer는 이득이 없다:

```text
personal vs global MSE 개선율:
  odd_even  lambda=10/25/50:  -1.3% / -0.1% / +0.6%(유의성 없음, p=0.80)
  chrono    lambda=10/25/50:  -7.7% / -5.1% / -3.2%  (모두 global보다 나쁨, p~0)
```

shrinkage를 키울수록 손해가 줄어드는 패턴은 개인 편차가 노이즈일 때의 전형적 신호다.

### 3. 대신 라벨에 큰 투수별 systematic offset이 있다

```text
투수별 평균 residual의 split-half 상관:  odd_even 0.92,  chrono 0.46-0.56
투수별 residual 변동성(sd)의 상관:       odd_even 0.63,  chrono 0.09-0.12
```

baseline이 실력을 제대로 추적한다면 투수별 평균 residual은 0 주변 노이즈여야 한다
(상관 ~0). 0.92는 `baseline_skill`이 특정 투수를 지속적으로 과대/과소평가한다는 뜻이다.
유력한 기제는 라벨 구성상의 shrinkage 편향이다: `baseline_skill`은 league/전시즌
prior 쪽으로(baseline_k=8), `shrunk_xwOBA`는 개인 prior 쪽으로(eb_k=10, BF~4에서
등판 실측 가중치 ~0.3) 당겨지는데, 두 축의 당김이 투수마다 다르게 어긋나 지속적
offset을 만든다. 그 결과 tertile 클래스의 일부는 "등판이 평소 대비 좋았나"가 아니라
"누구의 baseline이 편향돼 있나"를 인코딩한다.

## 판정

```text
반응 프로파일 개인화 (클러스터링/트랜스포머/GNN 공통 전제): 기각
  - 전제 (b) 실패: 개인 반응 편차의 시간 전이 = 0, transfer는 오히려 음(-)
  - 아키텍처를 바꿔도 학습할 안정적 개인 반응이 데이터에 없다

실제 개인화 문제의 위치: 모델이 아니라 라벨
  - 다음 작업 = 라벨 debias (prior-only 투수별 residual recentering)
  - debias 후 남는 안정 트레잇(residual 변동성)은 prior-only 피처로 활용 가능
```

## 다음 단계

1. `lib/labeling.py`에서 residual을 prior-only expanding 투수별 평균으로 recenter한
   `residual_centered`를 추가하고, tertile 컷과 학습 타깃을 그 기준으로 교체
2. debias 후 베이스라인 재실행 — 지표가 진짜 "본인 대비"를 재는지 확인
   (평균 residual split-half 상관이 0 근처로 떨어져야 debias 성공)
3. debias된 라벨 기준으로 워크로드 피처의 기여를 재평가 — 현재 lift 1.21이
   라벨 offset의 프록시(구속/구종 수준 피처)에서 온 것인지 분해

## 주의

- 선형 반응만 검정했다. 다만 반쪽당 n=25~100에서 선형 편차조차 전이가 안 되면
  더 많은 파라미터를 쓰는 비선형/시퀀스 개인화는 성립하기 더 어렵다.
- chrono 분할은 2023 피치클록 등 시대 변화와 겹친다. 시대 효과가 안정성 저하에
  기여했을 수 있으나, 실전 배치도 같은 시간 전이를 요구하므로 결론은 동일하다.
- 관측 데이터의 선택편향(감독이 지친 투수를 쉬게 함) 때문에 생리적 민감도가
  기용 패턴에 가려져 있을 수 있다. 이 결과는 "이 데이터에서 학습 불가"이지
  "개인차가 물리적으로 없다"가 아니다.
