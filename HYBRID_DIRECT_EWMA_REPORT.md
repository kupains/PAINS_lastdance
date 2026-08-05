# 공통·선수별 하이브리드 Stuff+ 예측 실험

생성 시각: 2026-08-05T15:00:14.246663+09:00

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

Ridge와 XGBoost는 각 공통 피처의 마지막값·평균·표준편차·추세와 EWMA4(총 53개)를 사용한다. TCN은 정규화된 8경기 시퀀스와 별도의 학습 가능한 EWMA4 입력을 사용한다.

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

- Ridge: EWMA4를 포함한 공통 Ridge 직접 예측 + 선수별 Ridge 보정
- XGBoost: EWMA4를 포함한 공통 부스팅 직접 예측 + 선수별 소형 부스팅 보정
- TCN: 공통 causal TCN 인코더 + 선수별 선형 보정층
- 최종값: 학습된 공통 Stuff+ 예측 + 선수별 보정. EWMA4의 계수도 모델이 학습한다.

계산식은 다음과 같다.

\[
\hat y_{i,t}=g(C_{i,t}, EWMA4_{i,t})+h_i(U_{i,t})
\]

Ridge에서는 `g`와 `h_i`가 선형식이고, XGBoost에서는 트리의 합, TCN에서는 causal encoder와 학습 가능한 EWMA skip 및 선수별 선형층이다. 이전 방식의 `EWMA4 + 0.1 × 잔차`는 사용하지 않는다.

## 6. 검증 선택

| model | candidate_id | mean_balanced_accuracy | selection_score |
| --- | --- | --- | --- |
| ridge | ridge_direct_ewma_g300_i300 | 0.4903 | 0.4681 |
| tcn | tcn_direct_ewma_std_warm_c4_b8_l20.5 | 0.5074 | 0.4911 |
| xgboost | xgb_direct_ewma_n100_d3_in20 | 0.5162 | 0.5062 |

## 7. 2025 결과

### 공통 표본

| model | n_total_eligible | n_predicted | coverage | n_evaluated | accuracy | ordinal_mae | low_precision | low_recall | low_f1 | true_low_count | predicted_low_count | predicted_low_rate | middle_precision | middle_recall | middle_f1 | true_middle_count | predicted_middle_count | predicted_middle_rate | high_precision | high_recall | high_f1 | true_high_count | predicted_high_count | predicted_high_rate | balanced_accuracy | macro_f1 | low_to_high_error_count | high_to_low_error_count | low_to_high_error_rate | high_to_low_error_rate | extreme_error_count | extreme_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ewma | 560 | 560 | 1.0000 | 560 | 0.5018 | 0.5571 | 0.6667 | 0.5639 | 0.6110 | 227 | 192 | 0.3429 | 0.3867 | 0.5266 | 0.4459 | 188 | 256 | 0.4571 | 0.4821 | 0.3724 | 0.4202 | 145 | 112 | 0.2000 | 0.4876 | 0.4924 | 17 | 16 | 0.0749 | 0.1103 | 33 | 0.0589 |
| hybrid_ridge | 560 | 560 | 1.0000 | 560 | 0.4839 | 0.5679 | 0.6524 | 0.5374 | 0.5894 | 227 | 187 | 0.3339 | 0.3816 | 0.6170 | 0.4715 | 188 | 304 | 0.5429 | 0.4783 | 0.2276 | 0.3084 | 145 | 69 | 0.1232 | 0.4607 | 0.4564 | 9 | 20 | 0.0396 | 0.1379 | 29 | 0.0518 |
| hybrid_xgboost | 560 | 560 | 1.0000 | 560 | 0.5036 | 0.5321 | 0.7063 | 0.4978 | 0.5840 | 227 | 160 | 0.2857 | 0.3971 | 0.7181 | 0.5114 | 188 | 340 | 0.6071 | 0.5667 | 0.2345 | 0.3317 | 145 | 60 | 0.1071 | 0.4835 | 0.4757 | 6 | 14 | 0.0264 | 0.0966 | 20 | 0.0357 |
| hybrid_tcn | 560 | 560 | 1.0000 | 560 | 0.4804 | 0.5500 | 0.6968 | 0.4758 | 0.5654 | 227 | 155 | 0.2768 | 0.3771 | 0.7021 | 0.4907 | 188 | 350 | 0.6250 | 0.5273 | 0.2000 | 0.2900 | 145 | 55 | 0.0982 | 0.4593 | 0.4487 | 3 | 14 | 0.0132 | 0.0966 | 17 | 0.0304 |

### 모델별 예측 가능 표본

| model | n_total_eligible | n_predicted | coverage | n_evaluated | accuracy | ordinal_mae | low_precision | low_recall | low_f1 | true_low_count | predicted_low_count | predicted_low_rate | middle_precision | middle_recall | middle_f1 | true_middle_count | predicted_middle_count | predicted_middle_rate | high_precision | high_recall | high_f1 | true_high_count | predicted_high_count | predicted_high_rate | balanced_accuracy | macro_f1 | low_to_high_error_count | high_to_low_error_count | low_to_high_error_rate | high_to_low_error_rate | extreme_error_count | extreme_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ewma | 560 | 560 | 1.0000 | 560 | 0.5018 | 0.5571 | 0.6667 | 0.5639 | 0.6110 | 227 | 192 | 0.3429 | 0.3867 | 0.5266 | 0.4459 | 188 | 256 | 0.4571 | 0.4821 | 0.3724 | 0.4202 | 145 | 112 | 0.2000 | 0.4876 | 0.4924 | 17 | 16 | 0.0749 | 0.1103 | 33 | 0.0589 |
| hybrid_ridge | 560 | 560 | 1.0000 | 560 | 0.4839 | 0.5679 | 0.6524 | 0.5374 | 0.5894 | 227 | 187 | 0.3339 | 0.3816 | 0.6170 | 0.4715 | 188 | 304 | 0.5429 | 0.4783 | 0.2276 | 0.3084 | 145 | 69 | 0.1232 | 0.4607 | 0.4564 | 9 | 20 | 0.0396 | 0.1379 | 29 | 0.0518 |
| hybrid_xgboost | 560 | 560 | 1.0000 | 560 | 0.5036 | 0.5321 | 0.7063 | 0.4978 | 0.5840 | 227 | 160 | 0.2857 | 0.3971 | 0.7181 | 0.5114 | 188 | 340 | 0.6071 | 0.5667 | 0.2345 | 0.3317 | 145 | 60 | 0.1071 | 0.4835 | 0.4757 | 6 | 14 | 0.0264 | 0.0966 | 20 | 0.0357 |
| hybrid_tcn | 560 | 560 | 1.0000 | 560 | 0.4804 | 0.5500 | 0.6968 | 0.4758 | 0.5654 | 227 | 155 | 0.2768 | 0.3771 | 0.7021 | 0.4907 | 188 | 350 | 0.6250 | 0.5273 | 0.2000 | 0.2900 | 145 | 55 | 0.0982 | 0.4593 | 0.4487 | 3 | 14 | 0.0132 | 0.0966 | 17 | 0.0304 |

### 연속형 Stuff+ 오차

| model | mae | rmse | continuous_n |
| --- | --- | --- | --- |
| ewma | 4.3213 | 5.6352 | 560 |
| hybrid_ridge | 4.3886 | 5.6161 | 560 |
| hybrid_xgboost | 4.2555 | 5.5013 | 560 |
| hybrid_tcn | 4.2742 | 5.4762 | 560 |

### 기존 잔차 방식과 직접예측 비교

| approach | model | accuracy | balanced_accuracy | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| ewma_plus_0.1_residual | ewma | 0.5018 | 0.4876 | 4.3213 | 5.6352 |
| ewma_plus_0.1_residual | hybrid_ridge | 0.5036 | 0.4894 | 4.2773 | 5.5698 |
| ewma_plus_0.1_residual | hybrid_xgboost | 0.5125 | 0.4988 | 4.3000 | 5.6041 |
| ewma_plus_0.1_residual | hybrid_tcn | 0.5089 | 0.4952 | 4.3062 | 5.6096 |
| direct_stuff_plus_with_ewma_feature | ewma | 0.5018 | 0.4876 | 4.3213 | 5.6352 |
| direct_stuff_plus_with_ewma_feature | hybrid_ridge | 0.4839 | 0.4607 | 4.3886 | 5.6161 |
| direct_stuff_plus_with_ewma_feature | hybrid_xgboost | 0.5036 | 0.4835 | 4.2555 | 5.5013 |
| direct_stuff_plus_with_ewma_feature | hybrid_tcn | 0.4804 | 0.4593 | 4.2742 | 5.4762 |

## 8. 선수별 마지막 예측

| row_id | pitcher | game_date | true_stuff_plus | true_class | q33 | q67 | pitcher_name | ewma_prediction | ewma_class | hybrid_ridge_prediction | hybrid_ridge_class | hybrid_xgboost_prediction | hybrid_xgboost_class | hybrid_tcn_prediction | hybrid_tcn_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 607d4aacc3d70f50c4ba8df2fef3179db47b5d997f8a47c92961f73f95f63a6c | 663903 | 2025-09-28 00:00:00 | 95.62 | 2 | 89.95 | 95.02 | Brady Singer | 98.22 | 2.00 | 95.10 | 2.00 | 96.82 | 2.00 | 94.67 | 1.00 |
| 291abfdfa34f946a18013f08f4a5bf3448d3d0f9e8e75e7a7705c9ddbf7ba8f9 | 450203 | 2025-09-13 00:00:00 | 95.56 | 0 | 98.54 | 104.56 | Charlie Morton | 96.24 | 0.00 | 96.69 | 0.00 | 96.54 | 0.00 | 98.06 | 0.00 |
| ff6ae8a3b7f40972b4c37f0111e5701351fe5cf47c72decbf634bccc350cca0b | 605135 | 2025-09-18 00:00:00 | 92.51 | 0 | 98.23 | 103.13 | Chris Bassitt | 95.03 | 0.00 | 97.26 | 0.00 | 97.19 | 0.00 | 98.06 | 0.00 |
| 240281594ca1a715e0b7499f9203ea336e397bbff58998195dc0e3357c025eac | 656302 | 2025-09-24 00:00:00 | 108.46 | 1 | 104.13 | 108.99 | Dylan Cease | 104.68 | 1.00 | 108.42 | 1.00 | 105.34 | 1.00 | 107.42 | 1.00 |
| a848e9c035e70baea4a5721b2c61b706c19d44e1b1deb0b07af837d38551cbd4 | 664285 | 2025-09-25 00:00:00 | 108.15 | 2 | 102.49 | 108.11 | Framber Valdez | 103.26 | 1.00 | 104.04 | 1.00 | 103.63 | 1.00 | 104.11 | 1.00 |
| f1596aa8ad34265140809f514196e358943f321b1210d78866c5bd2370f6959a | 592791 | 2025-09-27 00:00:00 | 92.59 | 0 | 94.83 | 98.84 | Jameson Taillon | 94.19 | 0.00 | 94.54 | 0.00 | 94.71 | 0.00 | 94.87 | 1.00 |
| 5e3b190148825a205a2f2effaae87ad6c5e0f0c9b807bc39d0da5b10f2e02b54 | 621244 | 2025-09-16 00:00:00 | 84.65 | 0 | 93.82 | 100.60 | José Berríos | 85.83 | 0.00 | 88.91 | 0.00 | 86.82 | 0.00 | 87.86 | 0.00 |
| 8865ba8be8e037d46119d4cd9c9e50e7ffa46bc66862252e658feccc04ad8cb6 | 592332 | 2025-09-28 00:00:00 | 97.95 | 1 | 97.31 | 104.02 | Kevin Gausman | 101.62 | 1.00 | 99.71 | 1.00 | 100.21 | 1.00 | 100.38 | 1.00 |
| 7f0fda03223443af5133d6ff6194f559e41d96cafd6136bf781edeaef8bf51a7 | 607536 | 2025-09-27 00:00:00 | 94.95 | 2 | 83.72 | 90.34 | Kyle Freeland | 92.76 | 2.00 | 90.65 | 2.00 | 91.69 | 2.00 | 92.38 | 2.00 |
| 55a57ca055e75c6511547bdfdf3fab5ef20179b8c738e26c057cb60a20af965c | 669302 | 2025-09-27 00:00:00 | 101.77 | 1 | 99.02 | 104.39 | Logan Gilbert | 103.95 | 1.00 | 100.06 | 1.00 | 102.27 | 1.00 | 101.19 | 1.00 |
| d01be132b93d2a839a6a983bf67bcfaf5a78e3a786ce2a7e6828024ec46d6e47 | 657277 | 2025-09-28 00:00:00 | 107.69 | 1 | 104.73 | 111.49 | Logan Webb | 107.40 | 1.00 | 108.94 | 1.00 | 109.16 | 1.00 | 107.71 | 1.00 |
| 2bf2cf7096775861e659eaddeb50d2091670f8e078cd0c457fa77e83259d0ac5 | 622491 | 2025-09-24 00:00:00 | 111.12 | 2 | 103.87 | 109.46 | Luis Castillo | 93.85 | 0.00 | 100.36 | 0.00 | 96.42 | 0.00 | 97.00 | 0.00 |
| c4171a3f7c9a8fdaed839e602a4242da98a3c5af958cd5d735cff7db8f7f711d | 608379 | 2025-09-27 00:00:00 | 90.78 | 0 | 94.94 | 99.94 | Michael Wacha | 92.04 | 0.00 | 95.82 | 1.00 | 94.08 | 0.00 | 95.04 | 1.00 |
| 9cae85c18c1eee0f68c69d1ff458a8d418a48151efb6d9c9e22ee92affaf9d6c | 656605 | 2025-09-26 00:00:00 | 95.36 | 0 | 96.27 | 103.79 | Mitch Keller | 92.75 | 0.00 | 95.63 | 0.00 | 94.80 | 0.00 | 95.89 | 0.00 |
| e8ba3eb7171720739c7339398c45be2d06f98de6d895ed8199798a160ee6329b | 571578 | 2025-09-28 00:00:00 | 93.28 | 1 | 87.83 | 93.32 | Patrick Corbin | 91.15 | 1.00 | 93.17 | 1.00 | 92.88 | 1.00 | 91.44 | 1.00 |
| 0ac9305c303f53868101ed31c58a25a5b2da718924be98700990c6bd0047a748 | 543243 | 2025-09-24 00:00:00 | 97.09 | 0 | 98.25 | 102.58 | Sonny Gray | 103.02 | 2.00 | 99.55 | 1.00 | 101.90 | 1.00 | 101.31 | 1.00 |
| 49b30f0e59e4e1194bb868a222254e8bbf3763dba0b9d1e46b21f5daa8e5623c | 542881 | 2025-08-29 00:00:00 | 105.47 | 2 | 99.37 | 103.54 | Tyler Anderson | 100.98 | 1.00 | 99.20 | 0.00 | 100.67 | 1.00 | 101.22 | 1.00 |
| 475a07340b479c6de20b884cb6cb77b161788e7a65dae6774631087bf3a18b61 | 668678 | 2025-09-26 00:00:00 | 96.04 | 0 | 96.51 | 102.55 | Zac Gallen | 96.92 | 1.00 | 95.50 | 0.00 | 96.30 | 0.00 | 96.45 | 0.00 |
| c8cbad4ca62b787664f1f6c2853da59f761d76f8a0270c248bc6cdf20ce8998b | 554430 | 2025-08-15 00:00:00 | 103.65 | 0 | 109.80 | 116.70 | Zack Wheeler | 106.91 | 0.00 | 104.98 | 0.00 | 108.63 | 0.00 | 107.92 | 0.00 |

## 9. 해석과 결론

공통 branch는 선수 간 반복되는 변화 패턴을 모으고, 선수별 branch는 같은 변화라도 선수마다 다른 반응을 보정한다. Ridge 계수 전체는 별도 CSV로 남겨 공통 효과와 선수별 효과를 분리해 확인할 수 있게 했다. 성능 판단은 2025 공통 표본의 balanced accuracy와 연속형 MAE를 함께 본다.

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
     공통 직접 Stuff+ 예측(EWMA 계수 학습) + 개인 보정
                    │
        연속 Stuff+ / 3단계 등급
```
