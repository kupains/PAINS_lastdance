# Stuff+ 통합 예측 보고서

생성: 2026-08-04T23:06:34+09:00

TCN은 제거된 모델이 아니다. 검증상 주 모델 교체 기준을 통과하지 못해 진단 모델로 남았지만, 고정 설정으로 2025 선수별 연속 예측까지 산출했다.

## 실행 계약

- 학습: 2020–2024
- 평가: 2025 (560경기)
- 공통 비교: 497경기
- 대상: 19명
- Ridge 전체 계수: 304개 (19명 × 16 features)

## 모델

- EWMA4: span=4
- Ridge: alpha=100, correction scale=0.25, 선수별 compact16
- XGBoost: trial 16, trees=100, depth=2, correction scale=0.2218450676
- TCN: H1 ordered, L=8, channels=4, bottleneck=4, alpha=0.1, epochs=250, seeds=[20260722, 20260723, 20260724, 20260725, 20260726]

## 산출물

전체 성능표, 선수별 최근 예측, Ridge 계수와 해석은 `FINAL_MODEL_REPORT.html`에 포함되어 있다. 원자료는 `experiments\runs\tcn_ridge_replay_2025`에 저장된다.
