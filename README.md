# MLB 선발투수 Stuff+ 범주 예측

리그 전체에서 2021~2025년 매 시즌 `공식 선발 + 50구 이상` 등판을 20경기 이상 확보한
19명을 대상으로 다음 선발 등판의 Stuff+를 선수별 Low/Middle/High 범주로 예측합니다.

## 채택 모델

- 비선형 주 예측: TPE로 최적화한 XGBoost + EWMA4
- 해석 보조: Ridge(`alpha=100`, correction scale 0.5) + EWMA4
- 하이퍼파라미터 탐색: 2022~2024 순차 검증만 사용
- 최종 재학습: 2020~2024
- 2025 진단 평가: XGBoost 51.71%, Ridge 52.92%

XGBoost 탐색은 32개 후보를 2023년에서 10개로, 2023~2024년에서 3개로 줄인 뒤
2022~2024년 안정성 점수로 잠급니다. 동일 seed 재실행에서 같은 파라미터가 나오는 것을
확인했으며 탐색 시간은 약 29초입니다.

## 원본 데이터 수집

아래 명령 하나가 데이터 선정부터 원본 수집과 감사 기록까지 수행합니다.

```powershell
.\.venv\Scripts\python.exe collect_mlb_stuff_dataset.py
.\.venv\Scripts\python.exe verify_mlb_stuff_collection.py
```

수집 순서는 다음과 같습니다.

1. MLB 공식 Stats API에서 2021~2025 정규시즌 투수 기록 전체를 연도별로 저장합니다.
2. 다섯 시즌 모두 20선발 이상인 MLBAM ID를 자동 선정합니다.
3. 선정된 전 선수의 2020~2025 Statcast 투구 원본을 선수별 parquet으로 저장합니다.
4. MLBAM ID를 FanGraphs ID로 변환하고 선발 경기 로그와 Stuff+를 JSON 원본 및 parquet으로 저장합니다.
5. `pitcher + game_date`로 매칭하고 50구 이상 조건을 적용해 최종 자격 선수를 계산합니다.

재현 및 감사용 파일은 git에 포함되는 `manifests/`에 생성됩니다.

- `stable_starter_candidates.csv`: 공식 기록 기준 최초 후보와 연도별 선발 수
- `collection_audit.csv`: 선수·연도별 Stuff+/Statcast 매칭 50구 이상 선발 수와 탈락 사유
- `mlb_stuff_collection_manifest.json`: 실행 명령, Git 커밋, 패키지 버전, 수집 범위,
  MLBAM↔FanGraphs 매핑, 모든 원본 파일의 SHA-256·행 수·날짜 범위

`data/reproducible_mlb_stuff/`의 원본은 용량 때문에 git에서 제외하지만 manifest의 해시로 동일 파일인지 확인할 수
있습니다. API를 다시 호출해 캐시까지 갱신하려면 `--force`를 사용합니다.

## XGBoost 탐색 실행

```powershell
python search_xgboost_fast.py `
  --statcast-dir data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2020 data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2020.parquet data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --official-stats data/reproducible_mlb_stuff/mlb_official_pitching_2021_2025.parquet `
  --output-dir experiments/runs/xgboost_fast_search_2020_2025
```

## 주요 탐색 산출물

- `stage1_32_candidates_2023.csv`
- `stage2_top10_2023_2024.csv`
- `stage3_top3_2022_2024.csv`
- `locked_xgboost_config.json`
- `locked_xgboost_2025_predictions.parquet`
- `ridge_benchmark_2025_predictions.parquet`
- `tuned_xgboost_vs_ridge_2025.parquet`
- `latest_player_predictions_2025.csv`: 선수별 최신 평가 등판의 실제 Stuff+, 두 모델의 연속 예측값과 범주

## EWMA 가중치 탐색

```powershell
python search_ewma_alpha.py `
  --statcast-dir data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2020 data/reproducible_mlb_stuff/statcast_mlb_stable_starters_2021_2025 `
  --stuff data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2020.parquet data/reproducible_mlb_stuff/fangraphs_stuff_mlb_stable_starters_2021_2025.parquet `
  --output-dir experiments/runs/ewma_alpha_search_2020_2025
```

2022~2024 순차 검증에서 전역 `alpha=0.05~0.95`를 탐색한 결과 `alpha=0.39`
(`span≈4.13`)가 안정성 포함 선택 점수 1위였다. 기존 `alpha=0.40`과의 차이는 매우 작았다.
세부 결과는 `EWMA_ALPHA_SEARCH_REPORT.md`에 정리돼 있습니다.

전체 방법론과 결과는 `FINAL_MODEL_REPORT.md`와 `FINAL_MODEL_REPORT.html`에 정리돼 있습니다.
선수별 최신 평가 등판의 연속 예측값은 `PLAYER_PREDICTIONS_2025.html`에서 확인할 수 있습니다.
