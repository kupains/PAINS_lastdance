# 공통·선수별 하이브리드 Stuff+ 예측 실험

생성 시각: 2026-08-05T15:21:18.119116+09:00

## 1. 개요

모든 선수에게 반복되는 상태·구종 구조는 공통 모델로 학습하고, 투구 위치와 구종별 세부 변화 등 나머지 정보는 선수별 보정 모델로 학습했다. 모델 선택은 2022~2024년에만 수행했고 2025년은 마지막 한 번의 고정 평가에 사용했다.

## 2. 데이터와 누수 방지

현재 경기의 Stuff+와 물리 피처는 입력하지 않았다. 각 예측 행은 경기일보다 엄격히 이전인 최대 8경기만 사용하며, 결측치 대치와 정규화 통계도 해당 fold의 훈련 구간에서 선수별로 계산했다.

## 3. 공통 피처

- `stuff_plus`
- `pitch_count`
- `rest_days_log`
- `long_gap`
- `season_start`
- `primary_fb_velocity`
- `primary_fb_spin`
- `primary_fb_pfx_x`
- `primary_fb_pfx_z`
- `arm_angle`
- `fastball_share`
- `breaking_share`
- `offspeed_share`

Ridge와 XGBoost는 각 공통 피처의 마지막값·평균·표준편차·추세와 선수 기준 EWMA z-score(총 53개)를 사용한다. TCN은 정규화된 8경기 시퀀스와 별도의 학습 가능한 EWMA z-score 입력을 사용한다.

## 4. 선수별 피처

- `release_speed`
- `effective_speed`
- `release_spin_rate`
- `release_extension`
- `release_pos_x`
- `release_pos_y`
- `release_pos_z`
- `spin_axis_sin`
- `spin_axis_cos`
- `pfx_x`
- `pfx_z`
- `plate_x`
- `plate_z`
- `zone`
- `api_break_z_with_gravity`
- `api_break_x_arm`
- `outing_xwOBA`
- `known_pitch_share`
- `ff_count`
- `ff_velocity`
- `ff_spin`
- `ff_pfx_x`
- `ff_pfx_z`
- `si_count`
- `si_velocity`
- `si_spin`
- `si_pfx_x`
- `si_pfx_z`
- `fa_count`
- `pitch_mix_CH`
- `pitch_mix_CS`
- `pitch_mix_CU`
- `pitch_mix_FC`
- `pitch_mix_FF`
- `pitch_mix_FS`
- `pitch_mix_KC`
- `pitch_mix_PO`
- `pitch_mix_SI`
- `pitch_mix_SL`
- `pitch_mix_ST`
- `pitch_mix_SV`
- `pitch_mix_UN`
- `prev_start_pitch_count`
- `rest_days`
- `workload_density_3starts`
- `prior_stuff_plus`
- `stuff_plus_mean_last5`
- `stuff_plus_slope_last5`
- `release_speed_ma5`
- `release_speed_slope5`
- `release_spin_rate_ma5`
- `release_spin_rate_slope5`
- `release_extension_ma5`
- `release_extension_slope5`
- `release_pos_x_ma5`
- `release_pos_x_slope5`
- `release_pos_z_ma5`
- `release_pos_z_slope5`
- `arm_angle_ma5`
- `arm_angle_slope5`
- `pfx_x_ma5`
- `pfx_x_slope5`
- `pfx_z_ma5`
- `pfx_z_slope5`
- `spin_axis_sin_ma5`
- `spin_axis_cos_ma5`
- `breaking_share_ma5`
- `breaking_share_slope5`
- `offspeed_share_ma5`
- `offspeed_share_slope5`
- `prior_minus_ewma4`
- `mean5_minus_ewma4`
- `rest_days_capped`
- `pitcher_prior_outing_count`
- `stuff_plus_lag1`
- `stuff_plus_prior_mean5`
- `stuff_plus_prior_std5`
- `stuff_plus_prior_slope5`
- `pitch_count_lag1`
- `pitch_count_prior_mean5`
- `fastball_share_prior_mean5`
- `breaking_share_prior_mean5`
- `release_extension_prior_mean5`
- `release_pos_x_prior_mean5`
- `release_pos_z_prior_mean5`
- `arm_angle_prior_mean5`

`release_pos_y`를 포함한 나머지 사용 가능 피처 전체를 과거 시퀀스 요약 및 사전 계산된 lag/slope 형태로 사용한다.

## 5. 모델 구조

- Ridge: 공통 player-z Ridge 예측 + 선수별 Ridge 보정
- XGBoost: 공통 player-z 부스팅 예측 + 선수별 소형 부스팅 보정
- TCN: 공통 causal TCN 인코더 + 선수별 선형 보정층
- 연속 목표: 선수 훈련 평균 대비 Stuff+ 표준점수

계산식은 다음과 같다.

\[
z_{i,t}=(y_{i,t}-\mu_i)/\sigma_i,\quad
\hat z_{i,t}=g(C_{i,t}, EWMAz_{i,t})+h_i(U_{i,t}),\quad
\hat y_{i,t}=\mu_i+\sigma_i\hat z_{i,t}
\]

등급은 `Low: z < -0.5`, `Middle: -0.5 ≤ z ≤ 0.5`, `High: z > 0.5`다. Ridge에서는 `g`와 `h_i`가 선형식이고, XGBoost에서는 트리의 합, TCN에서는 causal encoder와 학습 가능한 EWMA-z skip 및 선수별 선형층이다.

## 6. 검증 선택

| model | candidate_id | mean_balanced_accuracy | selection_score |
| --- | --- | --- | --- |
| ridge | ridge_player_z_g100_i300 | 0.4713 | 0.4357 |
| tcn | tcn_player_z_c4_b8_l20.5 | 0.4877 | 0.4662 |
| xgboost | xgb_player_z_n150_d2_in25 | 0.4243 | 0.4079 |

## 7. 2025 결과

### 공통 표본

| model | n_total_eligible | n_predicted | coverage | n_evaluated | accuracy | ordinal_mae | low_precision | low_recall | low_f1 | true_low_count | predicted_low_count | predicted_low_rate | middle_precision | middle_recall | middle_f1 | true_middle_count | predicted_middle_count | predicted_middle_rate | high_precision | high_recall | high_f1 | true_high_count | predicted_high_count | predicted_high_rate | balanced_accuracy | macro_f1 | low_to_high_error_count | high_to_low_error_count | low_to_high_error_rate | high_to_low_error_rate | extreme_error_count | extreme_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ewma | 560 | 560 | 1.0000 | 560 | 0.5143 | 0.5304 | 0.6667 | 0.5472 | 0.6010 | 212 | 174 | 0.3107 | 0.4371 | 0.5924 | 0.5030 | 211 | 286 | 0.5107 | 0.4700 | 0.3431 | 0.3966 | 137 | 100 | 0.1786 | 0.4942 | 0.5002 | 12 | 13 | 0.0566 | 0.0949 | 25 | 0.0446 |
| hybrid_ridge | 560 | 560 | 1.0000 | 560 | 0.5054 | 0.5250 | 0.6623 | 0.4717 | 0.5510 | 212 | 151 | 0.2696 | 0.4310 | 0.7251 | 0.5406 | 211 | 355 | 0.6339 | 0.5556 | 0.2190 | 0.3141 | 137 | 54 | 0.0964 | 0.4719 | 0.4686 | 4 | 13 | 0.0189 | 0.0949 | 17 | 0.0304 |
| hybrid_xgboost | 560 | 560 | 1.0000 | 560 | 0.4982 | 0.5304 | 0.6667 | 0.4434 | 0.5326 | 212 | 141 | 0.2518 | 0.4278 | 0.7583 | 0.5470 | 211 | 374 | 0.6679 | 0.5556 | 0.1825 | 0.2747 | 137 | 45 | 0.0804 | 0.4614 | 0.4514 | 4 | 12 | 0.0189 | 0.0876 | 16 | 0.0286 |
| hybrid_tcn | 560 | 560 | 1.0000 | 560 | 0.4893 | 0.5304 | 0.7040 | 0.4151 | 0.5223 | 212 | 125 | 0.2232 | 0.4179 | 0.7725 | 0.5424 | 211 | 390 | 0.6964 | 0.5111 | 0.1679 | 0.2527 | 137 | 45 | 0.0804 | 0.4518 | 0.4391 | 1 | 10 | 0.0047 | 0.0730 | 11 | 0.0196 |

### 모델별 예측 가능 표본

| model | n_total_eligible | n_predicted | coverage | n_evaluated | accuracy | ordinal_mae | low_precision | low_recall | low_f1 | true_low_count | predicted_low_count | predicted_low_rate | middle_precision | middle_recall | middle_f1 | true_middle_count | predicted_middle_count | predicted_middle_rate | high_precision | high_recall | high_f1 | true_high_count | predicted_high_count | predicted_high_rate | balanced_accuracy | macro_f1 | low_to_high_error_count | high_to_low_error_count | low_to_high_error_rate | high_to_low_error_rate | extreme_error_count | extreme_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ewma | 560 | 560 | 1.0000 | 560 | 0.5143 | 0.5304 | 0.6667 | 0.5472 | 0.6010 | 212 | 174 | 0.3107 | 0.4371 | 0.5924 | 0.5030 | 211 | 286 | 0.5107 | 0.4700 | 0.3431 | 0.3966 | 137 | 100 | 0.1786 | 0.4942 | 0.5002 | 12 | 13 | 0.0566 | 0.0949 | 25 | 0.0446 |
| hybrid_ridge | 560 | 560 | 1.0000 | 560 | 0.5054 | 0.5250 | 0.6623 | 0.4717 | 0.5510 | 212 | 151 | 0.2696 | 0.4310 | 0.7251 | 0.5406 | 211 | 355 | 0.6339 | 0.5556 | 0.2190 | 0.3141 | 137 | 54 | 0.0964 | 0.4719 | 0.4686 | 4 | 13 | 0.0189 | 0.0949 | 17 | 0.0304 |
| hybrid_xgboost | 560 | 560 | 1.0000 | 560 | 0.4982 | 0.5304 | 0.6667 | 0.4434 | 0.5326 | 212 | 141 | 0.2518 | 0.4278 | 0.7583 | 0.5470 | 211 | 374 | 0.6679 | 0.5556 | 0.1825 | 0.2747 | 137 | 45 | 0.0804 | 0.4614 | 0.4514 | 4 | 12 | 0.0189 | 0.0876 | 16 | 0.0286 |
| hybrid_tcn | 560 | 560 | 1.0000 | 560 | 0.4893 | 0.5304 | 0.7040 | 0.4151 | 0.5223 | 212 | 125 | 0.2232 | 0.4179 | 0.7725 | 0.5424 | 211 | 390 | 0.6964 | 0.5111 | 0.1679 | 0.2527 | 137 | 45 | 0.0804 | 0.4518 | 0.4391 | 1 | 10 | 0.0047 | 0.0730 | 11 | 0.0196 |

### 연속형 Stuff+ 오차

| model | mae | rmse | continuous_n | z_mae | z_rmse | z_correlation |
| --- | --- | --- | --- | --- | --- | --- |
| ewma | 4.3213 | 5.6352 | 560 | 0.6683 | 0.8629 | 0.5279 |
| hybrid_ridge | 4.4113 | 5.6436 | 560 | 0.6802 | 0.8620 | 0.4927 |
| hybrid_xgboost | 4.4033 | 5.6839 | 560 | 0.6777 | 0.8652 | 0.4886 |
| hybrid_tcn | 4.3056 | 5.5090 | 560 | 0.6635 | 0.8399 | 0.5308 |

### 기존 잔차 방식과 직접예측 비교

| approach | model | accuracy | balanced_accuracy | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| ewma_plus_0.1_residual_reclassified_at_half_sigma | ewma | 0.5143 | 0.4942 | 4.3213 | 5.6352 |
| ewma_plus_0.1_residual_reclassified_at_half_sigma | hybrid_ridge | 0.5232 | 0.4987 | 4.2773 | 5.5698 |
| ewma_plus_0.1_residual_reclassified_at_half_sigma | hybrid_xgboost | 0.5232 | 0.4996 | 4.3000 | 5.6041 |
| ewma_plus_0.1_residual_reclassified_at_half_sigma | hybrid_tcn | 0.5161 | 0.4932 | 4.3062 | 5.6096 |
| player_zscore_direct_with_ewma_z_feature | ewma | 0.5143 | 0.4942 | 4.3213 | 5.6352 |
| player_zscore_direct_with_ewma_z_feature | hybrid_ridge | 0.5054 | 0.4719 | 4.4113 | 5.6436 |
| player_zscore_direct_with_ewma_z_feature | hybrid_xgboost | 0.4982 | 0.4614 | 4.4033 | 5.6839 |
| player_zscore_direct_with_ewma_z_feature | hybrid_tcn | 0.4893 | 0.4518 | 4.3056 | 5.5090 |

## 8. 선수별 마지막 예측

| row_id | pitcher | game_date | true_stuff_plus | true_class | q33 | q67 | pitcher_name | ewma_prediction | ewma_z | ewma_class | hybrid_ridge_prediction | hybrid_ridge_z | hybrid_ridge_class | hybrid_xgboost_prediction | hybrid_xgboost_z | hybrid_xgboost_class | hybrid_tcn_prediction | hybrid_tcn_z | hybrid_tcn_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 607d4aacc3d70f50c4ba8df2fef3179db47b5d997f8a47c92961f73f95f63a6c | 663903 | 2025-09-28 00:00:00 | 95.62 | 2 | 89.95 | 95.02 | Brady Singer | 98.22 | 0.98 | 2 | 94.34 | 0.31 | 1 | 95.63 | 0.53 | 2 | 94.91 | 0.41 | 1 |
| 291abfdfa34f946a18013f08f4a5bf3448d3d0f9e8e75e7a7705c9ddbf7ba8f9 | 450203 | 2025-09-13 00:00:00 | 95.56 | 0 | 98.54 | 104.56 | Charlie Morton | 96.24 | -0.88 | 0 | 96.86 | -0.78 | 0 | 97.50 | -0.68 | 0 | 97.70 | -0.65 | 0 |
| ff6ae8a3b7f40972b4c37f0111e5701351fe5cf47c72decbf634bccc350cca0b | 605135 | 2025-09-18 00:00:00 | 92.51 | 0 | 98.23 | 103.13 | Chris Bassitt | 95.03 | -1.00 | 0 | 97.66 | -0.56 | 0 | 97.26 | -0.62 | 0 | 97.06 | -0.66 | 0 |
| 240281594ca1a715e0b7499f9203ea336e397bbff58998195dc0e3357c025eac | 656302 | 2025-09-24 00:00:00 | 108.46 | 1 | 104.13 | 108.99 | Dylan Cease | 104.68 | -0.30 | 1 | 107.76 | 0.17 | 1 | 106.33 | -0.05 | 1 | 106.77 | 0.02 | 1 |
| a848e9c035e70baea4a5721b2c61b706c19d44e1b1deb0b07af837d38551cbd4 | 664285 | 2025-09-25 00:00:00 | 108.15 | 1 | 102.49 | 108.11 | Framber Valdez | 103.26 | -0.33 | 1 | 104.06 | -0.22 | 1 | 102.15 | -0.49 | 1 | 104.26 | -0.19 | 1 |
| f1596aa8ad34265140809f514196e358943f321b1210d78866c5bd2370f6959a | 592791 | 2025-09-27 00:00:00 | 92.59 | 0 | 94.83 | 98.84 | Jameson Taillon | 94.19 | -0.56 | 0 | 94.24 | -0.55 | 0 | 93.56 | -0.68 | 0 | 94.42 | -0.51 | 0 |
| 5e3b190148825a205a2f2effaae87ad6c5e0f0c9b807bc39d0da5b10f2e02b54 | 621244 | 2025-09-16 00:00:00 | 84.65 | 0 | 93.82 | 100.60 | José Berríos | 85.83 | -1.51 | 0 | 88.93 | -1.14 | 0 | 91.34 | -0.85 | 0 | 88.43 | -1.20 | 0 |
| 8865ba8be8e037d46119d4cd9c9e50e7ffa46bc66862252e658feccc04ad8cb6 | 592332 | 2025-09-28 00:00:00 | 97.95 | 1 | 97.31 | 104.02 | Kevin Gausman | 101.62 | 0.08 | 1 | 100.12 | -0.13 | 1 | 101.44 | 0.06 | 1 | 100.58 | -0.06 | 1 |
| 7f0fda03223443af5133d6ff6194f559e41d96cafd6136bf781edeaef8bf51a7 | 607536 | 2025-09-27 00:00:00 | 94.95 | 2 | 83.72 | 90.34 | Kyle Freeland | 92.76 | 0.88 | 2 | 90.42 | 0.52 | 2 | 91.92 | 0.75 | 2 | 92.36 | 0.82 | 2 |
| 55a57ca055e75c6511547bdfdf3fab5ef20179b8c738e26c057cb60a20af965c | 669302 | 2025-09-27 00:00:00 | 101.77 | 1 | 99.02 | 104.39 | Logan Gilbert | 103.95 | 0.34 | 1 | 99.92 | -0.23 | 1 | 101.98 | 0.06 | 1 | 101.11 | -0.06 | 1 |
| d01be132b93d2a839a6a983bf67bcfaf5a78e3a786ce2a7e6828024ec46d6e47 | 657277 | 2025-09-28 00:00:00 | 107.69 | 1 | 104.73 | 111.49 | Logan Webb | 107.40 | -0.10 | 1 | 108.64 | 0.08 | 1 | 109.40 | 0.19 | 1 | 107.49 | -0.08 | 1 |
| 2bf2cf7096775861e659eaddeb50d2091670f8e078cd0c457fa77e83259d0ac5 | 622491 | 2025-09-24 00:00:00 | 111.12 | 2 | 103.87 | 109.46 | Luis Castillo | 93.85 | -1.80 | 0 | 98.66 | -1.11 | 0 | 100.01 | -0.92 | 0 | 97.70 | -1.25 | 0 |
| c4171a3f7c9a8fdaed839e602a4242da98a3c5af958cd5d735cff7db8f7f711d | 608379 | 2025-09-27 00:00:00 | 90.78 | 0 | 94.94 | 99.94 | Michael Wacha | 92.04 | -1.05 | 0 | 96.29 | -0.24 | 1 | 95.10 | -0.46 | 1 | 94.75 | -0.53 | 0 |
| 9cae85c18c1eee0f68c69d1ff458a8d418a48151efb6d9c9e22ee92affaf9d6c | 656605 | 2025-09-26 00:00:00 | 95.36 | 1 | 96.27 | 103.79 | Mitch Keller | 92.75 | -0.75 | 0 | 95.89 | -0.39 | 1 | 94.71 | -0.52 | 0 | 96.79 | -0.28 | 1 |
| e8ba3eb7171720739c7339398c45be2d06f98de6d895ed8199798a160ee6329b | 571578 | 2025-09-28 00:00:00 | 93.28 | 2 | 87.83 | 93.32 | Patrick Corbin | 91.15 | 0.17 | 1 | 92.92 | 0.48 | 1 | 91.52 | 0.23 | 1 | 91.23 | 0.18 | 1 |
| 0ac9305c303f53868101ed31c58a25a5b2da718924be98700990c6bd0047a748 | 543243 | 2025-09-24 00:00:00 | 97.09 | 0 | 98.25 | 102.58 | Sonny Gray | 103.02 | 0.47 | 1 | 98.84 | -0.28 | 1 | 100.95 | 0.10 | 1 | 101.40 | 0.18 | 1 |
| 49b30f0e59e4e1194bb868a222254e8bbf3763dba0b9d1e46b21f5daa8e5623c | 542881 | 2025-08-29 00:00:00 | 105.47 | 2 | 99.37 | 103.54 | Tyler Anderson | 100.98 | -0.04 | 1 | 99.74 | -0.30 | 1 | 100.21 | -0.20 | 1 | 100.96 | -0.05 | 1 |
| 475a07340b479c6de20b884cb6cb77b161788e7a65dae6774631087bf3a18b61 | 668678 | 2025-09-26 00:00:00 | 96.04 | 1 | 96.51 | 102.55 | Zac Gallen | 96.92 | -0.32 | 1 | 95.55 | -0.52 | 0 | 95.99 | -0.46 | 1 | 96.60 | -0.37 | 1 |
| c8cbad4ca62b787664f1f6c2853da59f761d76f8a0270c248bc6cdf20ce8998b | 554430 | 2025-08-15 00:00:00 | 103.65 | 0 | 109.80 | 116.70 | Zack Wheeler | 106.91 | -1.00 | 0 | 105.22 | -1.25 | 0 | 109.60 | -0.61 | 0 | 108.11 | -0.83 | 0 |

## 9. 해석과 결론

선수별 z-score는 등급 정의와 결과 표현에는 적합하지만, z 자체를 학습 목표로 둔 모델은 Middle로 수축해 기존 잔차 모델보다 낮았다. 따라서 최종 권장안은 기존 잔차 모델의 연속 Stuff+ 예측을 유지하고, 출력 단계에서 `predicted_z=(predicted Stuff+−선수 훈련평균)/선수 훈련표준편차`를 함께 제공하며 ±0.5σ로 등급을 파생하는 방식이다.

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
     공통 player-z 예측(EWMA-z 계수 학습) + 개인 보정
                    │
        연속 z 및 Stuff+ 복원 / ±0.5σ 3단계 등급
```
