# MLB 선발투수 Stuff+ 등급 및 연속값 예측 — 최종 보고서

작성: 2026-08-05T13:46+09:00
학습: 2020–2024 / 최종 평가: 2025 / 대상: 19명

## 1. 연구 개요

다음 선발 경기의 Stuff+ 연속값과 선수별 Low·Middle·High 등급을 예측했다. XGBoost·Ridge·TCN의 등급 성능은 비슷했으며, XGBoost는 비선형 관계, Ridge는 계수 해석, TCN은 100% 커버리지와 시계열 표현에서 역할이 나뉜다.

## 2. 데이터 수집

Statcast 투구 데이터, FanGraphs 등판 Stuff+, MLB 공식 선발 기록을 결합했다. `collect_mlb_stuff_dataset.py`가 수집 과정과 SHA-256 매니페스트를 남기고 `verify_mlb_stuff_collection.py`가 177개 파일, 19명, 모델 준비 2,600행을 검증한다.

## 3. 모델

- EWMA4: 최근 Stuff+ span-4 기준선
- Ridge: 선수별 compact16, alpha=100, correction scale=0.25
- XGBoost: 선수별 compact16, 100 trees, depth=2, correction scale=0.2218450676
- Shared TCN: 최근 8경기 raw15, H1 ordered, 5 seeds, 248 parameters
- 진단: Ordinal linear, Ridge–TCN ensemble

## 4. 결과와 해석

공통 497경기에서 XGBoost accuracy 50.10%, Ridge 50.10%, TCN 49.90%, EWMA4 49.09%였다. TCN은 전체 560경기를 모두 예측했고 연속 MAE 4.301로 가장 낮았다. Ridge와 XGBoost는 compact16 결측으로 497경기를 예측했다.

## 5. 결론

최근 흐름이 예측의 대부분을 설명하며 복잡한 모델의 추가 이득은 제한적이다. XGBoost를 비선형 주 모델, Ridge를 설명 모델, TCN을 시퀀스 진단 모델로 병행하는 구성이 현재 결과에 가장 적합하다.

## 6. 아키텍처와 파이프라인

전체 모델 구조, 데이터 수집부터 2025 평가와 HTML 생성까지의 파이프라인, 선수별 예측, Ridge 전체 304개 계수는 `FINAL_MODEL_REPORT.html`에 수록했다.
