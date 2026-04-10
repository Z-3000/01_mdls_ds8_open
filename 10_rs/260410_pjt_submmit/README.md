# MovieLens AutoInt+ 기반 영화 추천 시스템

## 0. 파일 설명 개요 (일부 파일은 용량문제로 github 에 업로드 되지 못했습니다.)
| 목적 | 파일 |
| --- | --- |
| AutoInt+ 구현 확인 | `streamlit/autoint.py` |
| 학습, 평가, 저장 전체 흐름 확인 | `260410_rs_pjt_03_train_tuned_reviewed.ipynb` |
| Streamlit 추천 시스템 확인 | `streamlit/show_st.py` |
| 실험 설계 표 | `model/baseline_tf_halfday/batch_plan_260410_0132.csv` |
| 다중 실험 결과 비교 | `model/baseline_tf_halfday/batch_results_260410_0132.csv` |

## 1. 평가 문항 대응표

### 1-1. 평가 문항 1: AutoInt+ 모델을 구현함

| 확인 항목 | 근거 |
| --- | --- |
| AutoInt+ 모델 구조 구현 | `streamlit/autoint.py`의 `FeaturesEmbedding`, `MultiHeadSelfAttention`, `MultiLayerPerceptron`, `AutoIntMLP`, `AutoIntMLPModel` |
| 모델 학습 코드 구현 | `260410_rs_pjt_03_train_tuned_reviewed.ipynb` |
| 모델 평가 코드 구현 | 같은 노트북의 분류 평가(`evaluate`) + 랭킹 평가(`evaluate_ranking`) |


### 1-2. 평가 문항 2: 구현한 추천 시스템을 시각화함

| 확인 항목 | 근거 |
| --- | --- |
| Streamlit 앱 구현 | `streamlit/show_st.py` |


실행 명령:

```bash
streamlit run streamlit/show_st.py
```

### 1-3. 평가 문항 3: AutoInt+ 모델 성능 향상을 위한 다양한 시도를 함 
- 개요 : 대규모 시도 보다는, 각 파라미터별 실험을 수행
- 목적 : 각 파라미터별 성능에 미치는 영향을 거시적 관점에서 파악 

| 확인 항목 | 근거 |
| --- | --- |
| 실험 설계표 존재 | `model/baseline_tf_halfday/batch_plan_260410_0132.csv` |
| 다중 실험 결과 비교 | `model/baseline_tf_halfday/batch_results_260410_0132.csv` |
`batch_size` |
| 실험 결과 정렬 기준 | `valid_ndcg_at_10` 중심으로 best run 선택 |

## 2. 저장소 구조
- 일부 파일은 용량문제로 github 에 업로드 되지 못했습니다.

```text
.
├── 260409_rs_pjt_01_eda.ipynb
├── 260409_rs_pjt_02_prepro.ipynb
├── 260410_rs_pjt_03_train_tuned_reviewed.ipynb
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│   ├── users.dat
│   ├── movies_prepro.csv
│   ├── ratings_prepro.csv
│   ├── users_prepro.csv
│   ├── movielens_rcmm_v1.csv
│   └── movielens_rcmm_v2.csv
├── helper/
│   ├── comenv.py
│   └── mltrack.py
├── model/
│   ├── baseline_tf_halfday/
│   │   ├── batch_plan_260410_0132.csv
│   │   ├── batch_results_260410_0132.csv
│   │   ├── summary/
│   │   └── E00_... ~ E14_.../
│   ├── baseline_tf/
│   ├── baseline_tf_tuned/
│   └── autoint_plus_batch/
├── streamlit/
│   ├── autoint.py
│   └── show_st.py
└── requirements.txt
```

## 3. 프로젝트 전체 흐름


```text
EDA
-> 전처리
-> 모델 입력 데이터 생성
-> AutoInt+ 모델 구현
-> 학습 / 검증 / 랭킹 평가
-> 실험 결과 저장
-> 최고 성능 모델 선택
-> Streamlit에서 추천 결과 시각화
```

| 단계 | 파일 | 역할 |
| --- | --- | --- |
| 데이터 탐색 | `260409_rs_pjt_01_eda.ipynb` | MovieLens 원본 데이터 분포 확인 |
| 전처리 | `260409_rs_pjt_02_prepro.ipynb` | 영화/사용자/평점 전처리 및 모델 입력용 테이블 생성 |
| 학습/튜닝 | `260410_rs_pjt_03_train_tuned_reviewed.ipynb` | AutoInt+ 구현, 학습, 평가, 저장, 실험 비교 |
| 추론 모듈 | `streamlit/autoint.py` | 저장된 모델 로드, 입력 인코딩, 점수 예측 |
| 서비스 UI | `streamlit/show_st.py` | Streamlit 기반 추천 결과 시각화 |

## 4. 학습 노트북 내용 

### 파일명 `260410_rs_pjt_03_train_tuned_reviewed.ipynb`

주요 섹션:

- `01_로컬 환경 설정`
- `02_Config (하이퍼파라미터 설정)`
- `03_모델 구성요소`
- `04_AutoIntMLP`
- `05_데이터셋 정리 및 분할`
- `06_학습 / 평가 함수`
- `07.MLflow 로깅 + run_experiment 보완`
- `08_실험 설계`
- `09_실행 준비`
- `10_실험 실행`
- `11_시각화 코드`


## 5. 데이터 구조와 입력 피처

### 5-1. 사용 데이터

- 원본 데이터: MovieLens 1M
- 전처리 결과: `data/movielens_rcmm_v2.csv`

### 5-2. 모델 입력 컬럼

모델 입력은 `CAT_COLS` 14개

| 컬럼명 | 의미 |
| --- | --- |
| `user_id` | 사용자 ID |
| `movie_id` | 영화 ID |
| `movie_decade` | 영화 출시 decade |
| `movie_year` | 영화 출시 연도 |
| `rating_year` | 평점 발생 연도 |
| `rating_month` | 평점 발생 월 |
| `rating_decade` | 평점 발생 decade |
| `genre1` | 주 장르 |
| `genre2` | 보조 장르 |
| `genre3` | 추가 장르 |
| `gender` | 사용자 성별 |
| `age` | 사용자 연령대 |
| `occupation` | 사용자 직업 |
| `zip` | 사용자 지역 코드 |

라벨 컬럼은 `label`

```text
14개 categorical feature + 1개 label
```

### 5-3. 실제 field dimension

학습에 사용된 `field_dims`

```text
[6035, 3705, 10, 81, 4, 12, 1, 18, 18, 16, 2, 7, 21, 3438]
```

필드별 차원수

| 필드 | field_dim |
| --- | ---: |
| `user_id` | 6035 |
| `movie_id` | 3705 |
| `movie_decade` | 10 |
| `movie_year` | 81 |
| `rating_year` | 4 |
| `rating_month` | 12 |
| `rating_decade` | 1 |
| `genre1` | 18 |
| `genre2` | 18 |
| `genre3` | 16 |
| `gender` | 2 |
| `age` | 7 |
| `occupation` | 21 |
| `zip` | 3438 |


## 6. AutoInt+ 모델 구조 설명


### 6-1. 구성 요소

| 클래스 | 역할 |
| --- | --- |
| `FeaturesEmbedding` | 각 categorical field를 embedding vector로 변환 |
| `MultiHeadSelfAttention` | field 간 interaction을 multi-head self-attention으로 학습 |
| `MultiLayerPerceptron` | embedding을 flatten한 뒤 비선형 결합 수행 |
| `AutoIntMLP` | attention path와 DNN path를 합쳐 최종 예측값 생성 |
| `AutoIntMLPModel` | Keras `Model` 래퍼 |

### 6-2. 구조적 특징

1. 입력은 `(batch_size, 14)` 형태의 정수 인코딩된 categorical index
2. `FeaturesEmbedding`을 거치면 `(batch_size, 14, embedding_size)`가 됨 
3. attention block은 이 텐서를 여러 번 통과시키며 field interaction을 학습합
4. attention 결과를 flatten해서 attention score 계산
5. 같은 embedding을 reshape해서 MLP score 계산 
6. 두 score를 더한 뒤 `sigmoid`를 적용해 최종 예측

### 6-3. anchor 기준 실제 shape 흐름

anchor 설정:

- `embedding_size = 16`
- `att_layer_num = 3`
- `att_head_num = 2`
- `dnn_hidden_units = (64, 64)`

shape 흐름:

```text
입력                  : (B, 14)
Embedding 출력        : (B, 14, 16)
Attention block x 3   : (B, 14, 16)
Flatten               : (B, 224)
Attention path score  : (B, 1)
DNN path input        : (B, 224)
DNN path score        : (B, 1)
최종 출력             : (B, 1)
```


## 7. 데이터 분할 및 학습/평가 전략

### 7-1. 데이터 분할 로직

학습 노트북은 `movielens_rcmm_v2.csv`를 아래 순서로 분할

1. positive(`label == 1`)가 3개 이상인 user만 선별 
2. 각 user의 마지막 positive 1개를 `test_pos_df`로 사용
3. test를 제외한 나머지 positive 중 마지막 1개를 `valid_pos_df`로 사용
4. 나머지를 `train_df`로 
5. `train_df`를 다시 stratified split 해서 `fit_df`, `val_df`로 나눔


### 7-2. 실제 분할 크기

| 객체 | row 수 |
| --- | ---: |
| 원본 CSV | 1,000,209 |
| `rank_df` | 999,983 |
| `train_df` | 987,913 |
| `fit_df` | 889,121 |
| `val_df` | 98,792 |
| `valid_pos_df` | 6,035 |
| `test_pos_df` | 6,035 |

정수 인코딩 후 shape:

| 객체 | shape |
| --- | --- |
| `X_fit` | `(889121, 14)` |
| `y_fit` | `(889121,)` |
| `X_val` | `(98792, 14)` |
| `y_val` | `(98792,)` |

### 7-3. 평가 방식

#### 분류 평가

- `BinaryCrossentropy`
- `BinaryAccuracy`
- `AUC`

즉 모델이 개별 샘플을 얼마나 잘 맞추는
#### 랭킹 평가

- `NDCG@10`
- `HitRate@10`

즉 추천 시스템 관점에서 top-k 추천 품질을 직접봄


## 8. 학습 파이프라인

### 8-1. Config 구성

모델 구조와 학습 설정은 dataclass로 관리됩니다.

- `ModelConfig`
  - `embedding_size`
  - `att_layer_num`
  - `att_head_num`
  - `att_res`
  - `dnn_hidden_units`
  - `dnn_dropout`
  - `l2_reg_embedding`
  - `l2_reg_dnn`
- `TrainConfig`
  - `learning_rate`
  - `batch_size`
  - `epochs`
  - `seed`
  - `early_stopping_patience`
  - `artifact_dir`
  - `mlflow_experiment`

### 8-2. 학습 순서

```text
CSV 로드
-> MovieLensBundle 생성
-> encoder_maps 학습
-> X_fit / y_fit, X_val / y_val 생성
-> tf.data.Dataset 생성
-> AutoIntMLPModel build/compile
-> fit
-> classification 평가
-> ranking 평가
-> artifact 저장
-> MLflow 및 summary CSV 기록
```

### 8-3. compile 설정

- optimizer: `Adam`
- loss: `BinaryCrossentropy`
- metrics: `acc`, `auc`

### 8-4. artifact 저장 파일

| 파일 | 의미 |
| --- | --- |
| `config.json` | 모델/학습 설정과 feature 정보 |
| `encoder_maps.json` | category -> index 인코딩 맵 |
| `field_dims.npy` | 각 필드의 category 수 |
| `model.weights.h5` | 학습된 모델 가중치 |
| `metrics.json` | 최종 성능 지표 |
| `train_history.csv` | epoch별 학습 이력 |
| `valid_user_metrics.csv` | validation user 단위 ranking 결과 |
| `test_user_metrics.csv` | test user 단위 ranking 결과 |
| `run_context.json` | 실험 태그, 추가 메타데이터 |

즉 "모델을 저장했다"는 평가 문항은 단순히 가중치 하나만 남긴 것이 아니라,  
재현에 필요한 설정과 encoder까지 함께 저장했다는 점에서 더 명확하게 충족합니다.

## 9. 하이퍼파라미터 실험 설계

### 9-1. 기준(anchor) 설정

| 항목 | 값 |
| --- | --- |
| `embedding_size` | 16 |
| `att_layer_num` | 3 |
| `att_head_num` | 2 |
| `dnn_hidden_units` | `(64, 64)` |
| `dnn_dropout` | 0.4 |
| `learning_rate` | 0.001 |
| `batch_size` | 2048 |
| `epochs` | 8 |

anchor run:

```text
E00_anchor__autoint_emb16_attL3_attH2_dnn64x64_lr0p001_bs2048
```

### 9-2. 이번 배치에서 바꾼 값

`model/baseline_tf_halfday/batch_plan_260410_0132.csv` 기준 총 15개 실험이 실행

| 그룹 | 실험 예시 | 바꾼 값 |
| --- | --- | --- |
| anchor | `E00_anchor` | 기준 조합 |
| embedding/lr | `E01` ~ `E05` | `embedding_size`, `learning_rate` |
| learning rate | `E06` | `learning_rate` |
| attention layer | `E07`, `E08` | `att_layer_num` |
| attention head | `E09`, `E10` | `att_head_num` |
| DNN 구조 | `E11`, `E12` | `dnn_hidden_units` |
| batch size | `E13`, `E14` | `batch_size` |


### 9-3. 실험 결과 정렬 기준

최종 실험 결과는 다음 우선순위로 정렬
1. `valid_ndcg_at_10`
2. `test_ndcg_at_10`
3. `valid_auc`


## 10. 저장된 실험 결과 해석

실험 결과 파일:

```text
model/baseline_tf_halfday/batch_results_260410_0132.csv
```

### 10-1. top run 요약

| 구분 | `exp_id` | 핵심 변경 | `valid_ndcg@10` | `test_ndcg@10` | `valid_hr@10` | `test_hr@10` | `valid_auc` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| anchor | `E00_anchor` | 기준 조합 | 0.1151 | 0.0836 | 0.2583 | 0.1795 | 0.8114 |
| best ranking | `E07_attL1` | `att_layer_num: 3 -> 1` | 0.1883 | 0.1095 | 0.3881 | 0.2258 | 0.8074 |
| best low-lr | `E03_emb16_lr5e4` | `learning_rate: 1e-3 -> 5e-4` | 0.1610 | 0.1055 | 0.3231 | 0.2209 | 0.8086 |
| compact DNN | `E11_dnn32x32` | `dnn_hidden_units: (64,64) -> (32,32)` | 0.1492 | 0.1070 | 0.3180 | 0.2263 | 0.8116 |
| best AUC | `E06_lr2e3` | `learning_rate: 1e-3 -> 2e-3` | 0.1465 | 0.0964 | 0.3069 | 0.2169 | 0.8159 |

### 10-2. anchor 대비 최고 성능 run 개선폭

최고 성능 run:

```text
E07_attL1__autoint_emb16_attL1_attH2_dnn64x64_lr0p001_bs2048
```

anchor 대비 변화:

| 지표 | anchor | best(`E07_attL1`) | 개선 |
| --- | ---: | ---: | ---: |
| `valid_ndcg@10` | 0.1151 | 0.1883 | +0.0732 |
| `test_ndcg@10` | 0.0836 | 0.1095 | +0.0259 |
| `valid_hitrate@10` | 0.2583 | 0.3881 | +0.1297 |
| `test_hitrate@10` | 0.1795 | 0.2258 | +0.0464 |

해석:

- attention layer를 깊게 쌓는 것이 무조건 좋은 것은 아님
- 오히려 `att_layer_num=1`이 top-k 추천 품질에서 가장 좋았음 

### 10-3. 파라미터별 관찰 포인트

#### attention depth

- `att_layer_num=1`인 `E07_attL1`이 전체 1등
- `att_layer_num=4`인 `E08_attL4`는 AUC는 높지만 ranking은 `E07`보다 낮음

#### DNN hidden size

- `(32, 32)`가 `(64, 64)` anchor보다 ranking에서 개선
- 더 큰 `(128, 64)`는 오히려 성능 향상으로 이어지지 않은것 같음 


## 11. Streamlit 추천 시스템 구조

### 11-1. 동작 개요

Streamlit 앱은 저장된 최고 성능 모델을 불러와, 사용자가 아직 보지 않은 영화 중 상위 10개를 추천

동작 순서:

```text
batch_results 파일 읽기
-> 최고 성능 artifact 선택
-> 모델 / encoder 로드
-> 사용자 입력 받기
-> 사용자가 안 본 영화 후보군 생성
-> 각 후보 영화 점수 예측
-> top 10 추천 결과 출력
```
## 12. 결과 산출물 정리

### 12-1. 실험 계획 및 결과

- 실험 계획표: `model/baseline_tf_halfday/batch_plan_260410_0132.csv`
- 통합 결과표: `model/baseline_tf_halfday/batch_results_260410_0132.csv`

### 12-2. 시각화 결과

`model/baseline_tf_halfday/summary/`

- `embedding_lr_heatmap.png`
- `param_impact_grid.png`
- `raw_grid_emb_lr.png`
- `raw_single_factor_panels.png`
- `raw_top_runs_barplot.png`
- `top_runs_barplot.png`

### 12-3. 최고 성능 모델 artifact

`model/baseline_tf_halfday/E07_attL1__autoint_emb16_attL1_attH2_dnn64x64_lr0p001_bs2048/`

## 13. 코드 이해 포인트 요약

1. 모델 입력은 문자열이 아니라 정수 인코딩된 categorical index
2. `encoder_maps.json`이 학습과 추론의 일관성을 보장
3. AutoInt+의 핵심은 attention path와 DNN path를 함께 쓰는 구조
4. 분류 성능과 추천 순위 성능은 다를 수 있으므로, `NDCG@10`과 `HitRate@10`을 따로 봐야 함

## 14. 향후 개선 아이디어
1. `att_layer_num=1` 근방에서 더 촘촘한 탐색
2. `learning_rate=5e-4 ~ 1e-3` 구간 재탐색
3. `dnn_hidden_units=(32,32)`와 `(64,64)` 사이 추가 실험
4. ranking loss 또는 pairwise loss 도입 검토
5. Streamlit 화면에 추천 score, 장르 설명, 추천 근거 시각화 추가
