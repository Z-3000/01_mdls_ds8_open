# DKTC 위협 대화 분류 모델 개선 런북

작성일: 2026-03-17  
기준 세션: DKTC 데이터 전처리, `klue/roberta-base` 하이퍼파라미터 실험, Focal Loss 실험까지 완료된 상태

---

## 1. 문서 목적

이 문서는 이번 세션에서 진행한 DKTC 위협 대화 분류 모델 개선 작업을 세션 종료 이후에도 이어갈 수 있도록 정리한 운영 문서다.

정리 범위는 다음과 같다.

1. 사용한 데이터와 전처리 파이프라인
2. 시도한 모델/손실함수/하이퍼파라미터 실험
3. 실험별 성능 변화
4. 현재 병목 해석
5. 바로 이어서 해야 할 실험 우선순위
6. 외부 벤치마크 사례와 현재 과제에 주는 시사점

본 문서는 **이번 세션에서 실제로 확인된 값**과 **공식/공개 자료에서 확인 가능한 벤치마크 사례**만 사용했다. 확정할 수 없는 부분은 추정으로 단정하지 않았다.

---

## 2. 현재 작업 환경

- 모델 백본: `klue/roberta-base`
- 과제: DKTC 5-class 분류
  - 협박
  - 갈취
  - 직장내괴롭힘
  - 기타괴롭힘
  - 일반대화
- 최대 길이: `max_length=512`
- 장비: RTX 5060 Ti 16GB, CUDA 사용 가능
- 주요 라이브러리 버전
  - transformers 5.3.0
  - datasets 4.8.2
  - evaluate 0.4.6
  - accelerate 1.13.0
  - PyTorch 2.10.0+cu128

근거: `roberta-base_hp.ipynb`, `roberta-base_hp_Focal Loss.ipynb` 출력 로그

---

## 3. 데이터 구성과 전처리 이력

### 3.1 원천 데이터

#### DKTC
- 총 3,950개
- 클래스 분포
  - 기타 괴롭힘 대화: 1,094
  - 갈취 대화: 981
  - 직장 내 괴롭힘 대화: 979
  - 협박 대화: 896

#### 020 일반대화 데이터
- 전체 87,331개 중 자연 필터 후 39,802개
- 여기서 1,000개 샘플링하여 일반대화 클래스로 사용

근거: `merge_preprocessing_pipeline.ipynb`

### 3.2 병합 후 데이터

- DKTC 3,950 + 일반대화 1,000 = 4,950개
- 병합 직후 클래스 분포
  - 기타 괴롭힘 대화: 1,094
  - 일반 대화: 1,000
  - 갈취 대화: 981
  - 직장 내 괴롭힘 대화: 979
  - 협박 대화: 896

### 3.3 정제 및 이상치 필터링

정제 후 클래스별 평균 통계:
- 갈취: 평균 turn 10.5 / 평균 글자수 204.7
- 기타 괴롭힘: 평균 turn 10.2 / 평균 글자수 199.0
- 일반 대화: 평균 turn 11.6 / 평균 글자수 228.5
- 직장 내 괴롭힘: 평균 turn 10.4 / 평균 글자수 226.2
- 협박: 평균 turn 10.3 / 평균 글자수 234.6

이상치 필터링 후:
- 4,950 → 4,845 (105개 제거)
- 클래스 분포
  - 기타 괴롭힘 대화: 1,010
  - 일반 대화: 1,000
  - 갈취 대화: 973
  - 직장 내 괴롭힘 대화: 970
  - 협박 대화: 892

### 3.4 train/validation split 및 균형 조정

- 분할 후
  - 학습: 3,876
  - 검증: 969

- 증강/균형 조정 목표: 클래스당 778개 수준
- 최종 학습셋: 3,942개
- 최종 검증셋: 969개

최종 학습 클래스 분포:
- 기타 괴롭힘 대화: 808
- 일반 대화: 800
- 협박 대화: 778
- 갈취 대화: 778
- 직장 내 괴롭힘 대화: 778

최종 검증 클래스 분포:
- 기타 괴롭힘 대화: 202
- 일반 대화: 200
- 갈취 대화: 195
- 직장 내 괴롭힘 대화: 194
- 협박 대화: 178

### 3.5 길이 기반 설정 판단

품질 리포트 기준:
- 최대 길이: 918자
- 95퍼센타일: 451자
- `max_length=512`일 때 커버리지: **97.1%**

따라서 `max_length=512` 선택은 현재 데이터 기준으로 타당하다.

---

## 4. 베이스라인 하이퍼파라미터 실험 요약 (CrossEntropy/Weighted Loss 계열)

### 4.1 실험 조건

원본 베이스라인 설정:
- learning_rate = 2e-5
- epochs = 3
- batch_size = 16
- weight_decay = 0.01
- scheduler = linear
- warmup_ratio = 0.1
- weighted_loss = True

총 10개 실험 수행.

### 4.2 전체 결과

| 순위 | 실험명 | Macro F1 | Acc | Weighted F1 | Baseline 대비 |
|---|---:|---:|---:|---:|---:|
| 1 | wd_0.1 | 0.9265 | 0.9267 | 0.9271 | +0.0118 |
| 2 | wd_0 | 0.9249 | 0.9257 | 0.9256 | +0.0102 |
| 3 | smooth_0.1 | 0.9232 | 0.9236 | 0.9239 | +0.0085 |
| 4 | no_weight | 0.9232 | 0.9236 | 0.9239 | +0.0085 |
| 5 | cosine | 0.9217 | 0.9226 | 0.9225 | +0.0070 |
| 6 | epoch_5 | 0.9216 | 0.9216 | 0.9218 | +0.0069 |
| 7 | no_warmup | 0.9201 | 0.9205 | 0.9207 | +0.0054 |
| 8 | lr_5e-5 | 0.9185 | 0.9185 | 0.9188 | +0.0038 |
| 9 | baseline | 0.9147 | 0.9143 | 0.9150 | +0.0000 |
| 10 | lr_1e-5 | 0.9139 | 0.9143 | 0.9146 | -0.0008 |

근거: `roberta-base_hp.ipynb` 결과표

### 4.3 best 모델 상세

Best: `wd_0.1`
- Accuracy: 0.9267
- Macro F1: 0.9265
- Weighted F1: 0.9271

클래스별 F1:
- 협박: 0.8934
- 갈취: 0.8990
- 직장내괴롭힘: 0.9606
- 기타괴롭힘: 0.8894
- 일반대화: 0.9899

오분류 71 / 969 = 7.3%

주요 오분류 패턴:
- 협박 → 기타괴롭힘: 12
- 기타괴롭힘 → 갈취: 11
- 협박 → 갈취: 10
- 갈취 → 협박: 9
- 직장내괴롭힘 → 기타괴롭힘: 9

### 4.4 해석

1. **weight decay 0.1이 가장 효과적이었다.**
2. **label smoothing과 weighted loss 제거는 큰 차이를 만들지 못했다.**
3. **일반대화는 이미 매우 잘 맞는다.**
4. 병목은 `일반대화 vs 괴롭힘`보다 **괴롭힘 내부 클래스 경계**다.
5. 특히 아래 축이 핵심 혼동축이다.
   - 협박 ↔ 갈취
   - 협박 ↔ 기타괴롭힘
   - 기타괴롭힘 ↔ 갈취
   - 직장내괴롭힘 ↔ 기타괴롭힘

---

## 5. Focal Loss 실험 요약

### 5.1 실험 목적

가설:
- 일반대화는 상대적으로 쉬운 샘플일 가능성이 높다.
- 협박/갈취/기타괴롭힘은 경계가 애매한 hard sample이 많다.
- 따라서 Focal Loss를 사용하면 쉬운 샘플 기여를 줄이고 hard sample 학습을 강화할 수 있다.

### 5.2 실험 조건

기본 focal 설정:
- gamma = 2.0
- class weight 사용 여부 비교
- learning rate / epoch / weight decay / warmup / scheduler 실험

총 11개 실험 수행.

### 5.3 전체 결과

| 순위 | 실험명 | Macro F1 | Acc | Weighted F1 | Focal baseline 대비 |
|---|---:|---:|---:|---:|---:|
| 1 | focal_epoch_5 | 0.9301 | 0.9298 | 0.9302 | +0.0125 |
| 2 | focal_lr_5e-5 | 0.9272 | 0.9278 | 0.9278 | +0.0096 |
| 3 | focal_cosine | 0.9250 | 0.9257 | 0.9257 | +0.0074 |
| 4 | focal_no_warmup | 0.9245 | 0.9247 | 0.9249 | +0.0069 |
| 5 | focal_wd_0 | 0.9239 | 0.9247 | 0.9246 | +0.0063 |
| 6 | focal_wd_0.1 | 0.9233 | 0.9236 | 0.9240 | +0.0057 |
| 7 | focal_no_cw | 0.9220 | 0.9226 | 0.9227 | +0.0044 |
| 8 | focal_g1.5 | 0.9217 | 0.9226 | 0.9225 | +0.0041 |
| 9 | focal_lr_1e-5 | 0.9209 | 0.9216 | 0.9217 | +0.0033 |
| 10 | focal_g2.5 | 0.9203 | 0.9205 | 0.9209 | +0.0027 |
| 11 | focal_baseline | 0.9176 | 0.9174 | 0.9180 | +0.0000 |

근거: `roberta-base_hp_Focal Loss.ipynb` 결과표

### 5.4 best 모델 상세

Best: `focal_epoch_5`
- Accuracy: 0.9298
- Macro F1: 0.9301
- Weighted F1: 0.9302

클래스별 F1:
- 협박: 0.9186
- 갈취: 0.8990
- 직장내괴롭힘: 0.9634
- 기타괴롭힘: 0.8794
- 일반대화: 0.9899

오분류 68 / 969 = 7.0%

주요 오분류 패턴:
- 기타괴롭힘 → 갈취: 20
- 협박 → 갈취: 11
- 협박 → 기타괴롭힘: 8
- 직장내괴롭힘 → 기타괴롭힘: 7
- 기타괴롭힘 → 협박: 5

### 5.5 sanity check

- 검증셋 969개에서 오분류 68건이면 정분류 901건
- `901 / 969 = 0.9298`로 reported accuracy와 일치
- 클래스별 F1 평균도 약 0.9301로 macro F1과 일치

### 5.6 해석

1. **Focal Loss 실험 공간의 최고점은 non-focal 최고점보다 높다.**
   - non-focal best: 0.9265
   - focal best: 0.9301
   - 차이: **+0.0036 macro F1**

2. 하지만 개선 원인을 단순히 "Focal Loss 때문"이라고 단정하기는 어렵다.
   - epoch 5의 기여가 크다.
   - lr 5e-5, cosine도 유효했다.
   - 즉 **loss와 학습 스케줄 조합 효과**로 보는 것이 더 정확하다.

3. Focal Loss를 써도 병목은 그대로다.
   - 일반대화는 여전히 강함
   - 가장 약한 클래스는 여전히 `기타괴롭힘`
   - 핵심 혼동축도 계속 유지됨

4. `gamma` 튜닝은 1순위 변수가 아니었다.
   - gamma 1.5, 2.0, 2.5 간 큰 차이 없음
   - 현재 근거상 `1.5~2.0` 범위면 충분

5. class weight 효과는 미미하거나 불리할 가능성이 있다.
   - focal baseline(가중치 사용): 0.9176
   - focal_no_cw(가중치 미사용): 0.9220
   - 단, 단일 split 결과이므로 확정적 결론은 아님

---

## 6. 현재까지의 종합 판단

### 6.1 확실한 것

- `klue/roberta-base`는 현재 데이터에서 이미 강한 베이스라인이다.
- 데이터 전처리와 균형 조정은 일정 수준 이상 잘 작동했다.
- 일반대화 분류는 이미 매우 강하다.
- 진짜 병목은 괴롭힘 내부 클래스 경계다.
- 특히 `기타괴롭힘`이 가장 어려운 클래스다.

### 6.2 확실하지 않은 것

- Focal Loss가 public/private leaderboard에서도 항상 더 낫다고 단정할 수는 없다.
- 현재 결과는 단일 validation split 기준이다.
- Kaggle test 분포가 다르면 `일반대화` 비중 변화에 따라 모델 유불리가 바뀔 수 있다.

### 6.3 현재 최선의 해석

지금 단계에서 모델 개선은 **더 큰 backbone 탐색**보다 아래 축이 더 중요하다.

1. 괴롭힘 내부 클래스 경계 강화
2. 추론 시 경계 샘플 후처리
3. 여러 성향 모델 준비 후 leaderboard 반응 확인

---

## 7. 지금까지 시도한 것과 성능 변화 요약

### 7.1 데이터 측면

- 일반대화 외부 데이터 1,000개 추가
- 텍스트 정제 및 이상치 제거
- 클래스 균형 조정
- `max_length=512`로 토큰 길이 설정

### 7.2 모델 측면

#### non-focal 실험
- learning rate: 1e-5, 2e-5, 5e-5
- epoch: 3 vs 5
- weight decay: 0.0 / 0.01 / 0.1
- scheduler: linear / cosine
- warmup: 0.0 / 0.1
- label smoothing: 0.0 / 0.1
- weighted loss on/off

#### focal 실험
- gamma: 1.5 / 2.0 / 2.5
- class weight on/off
- learning rate, epoch, wd, scheduler, warmup 조합 반복

### 7.3 수치상 가장 중요한 변화

- baseline: **0.9147**
- non-focal best (`wd_0.1`): **0.9265**
  - baseline 대비 **+0.0118**
- focal best (`focal_epoch_5`): **0.9301**
  - focal baseline 대비 **+0.0125**
  - original baseline 대비 **+0.0154**

즉 이번 세션 전체 기준 최종 최고점은 **baseline 대비 +0.0154 macro F1** 개선이다.

---

## 8. 다음 세션에서 바로 이어서 할 일

우선순위는 아래 순서가 맞다.

### 8.1 1순위 — 최고 3개 후보 모델 보존

즉시 후보로 유지할 모델:
1. `focal_epoch_5`
2. `focal_lr_5e-5`
3. `focal_cosine`

이유:
- 성능 상위권
- 서로 성향이 다름
- Kaggle public/private 분포 차이에 대응하기 좋음

### 8.2 2순위 — 규칙 기반 후처리(logit adjustment)

목표:
- `협박 ↔ 갈취`
- `기타괴롭힘 ↔ 갈취`
- `직장내괴롭힘 ↔ 기타괴롭힘`
혼동 축 완화

추천 방식:
- 학습 데이터는 그대로 유지
- 추론 시 top1-top2 margin이 작을 때만 규칙 점수 추가
- 예시 규칙
  - 협박 신호: 해악 예고 표현
  - 갈취 신호: 요구/대가/금전/행동 강요 표현
  - 직장내괴롭힘 신호: 회사/상사/업무/평가/보고 문맥

### 8.3 3순위 — 2단계 분류 실험

1단계:
- 일반대화 vs 괴롭힘

2단계:
- 협박 / 갈취 / 직장내괴롭힘 / 기타괴롭힘

현재 결과상 일반대화는 이미 잘 맞으므로,
문제를 두 단계로 나눠 괴롭힘 내부 분리를 더 집중시키는 전략이 타당하다.

### 8.4 4순위 — CV 안정화

현재 결과는 holdout 기반이다.
다음 세션에서는 최소 3-fold stratified validation으로 재검증하는 것이 좋다.

왜 필요한가:
- class weight 효과가 실제인지 확인
- focal 개선이 split-specific인지 확인
- Kaggle public overfitting 방지

### 8.5 5순위 — 데이터 재설계

지금 단계에서는 **일반대화를 무조건 더 늘리는 것**이 1순위가 아니다.
오히려 아래가 더 중요하다.

- 기타괴롭힘 경계 샘플 확보
- 갈취/협박을 헷갈리게 만드는 패턴 확인
- 직장 맥락이 있는 괴롭힘과 없는 괴롭힘 분리 강화

---

## 9. 개선 전략 제안

### 전략 A. 단기 전략 (대회 직전 / 1일 대회 대응)

가장 현실적인 전략이다.

1. `focal_epoch_5`, `focal_lr_5e-5`, `focal_cosine` 저장
2. 규칙 기반 후처리 1개 추가
3. 대회 시작 후 2~3개 후보 제출
4. public leaderboard 반응을 보고 모델 성향 판단
5. public에만 과적합하지 않도록 여러 성향 모델 유지

### 전략 B. 중기 전략 (세션 이어서 개선)

1. 3-fold 이상 CV 도입
2. 2단계 분류 파이프라인 실험
3. 혼동축 기준 규칙/후처리 강화
4. error analysis 문서화

### 전략 C. 장기 전략 (시간이 더 있을 때)

1. in-domain MLM 추가 사전학습
2. 다른 한국어 backbone 1~2개 비교
3. seed ensemble / model ensemble
4. pseudo-labeling 제한적 도입

---

## 10. 벤치마크 사례와 시사점

### 10.1 OffensEval 2020 — 도메인 적응 MLM + Transformer

UHH-LT 팀은 OffensEval 2020에서 **in-domain MLM fine-tuning을 거친 RoBERTa 기반 분류기**로 영어 트랙 1위를 보고했다. 핵심 포인트는 단순 supervised fine-tuning만이 아니라, **실제 과제와 유사한 도메인 데이터로 MLM 추가 사전학습을 수행한 뒤 분류에 투입했다는 점**이다.

시사점:
- 현재 DKTC도 일반 한국어 문장 분류가 아니라 괴롭힘/위협 대화라는 도메인 특성이 강하다.
- 시간이 더 있다면 train+unlabeled text 기반의 domain adaptive pretraining(MLM)을 검토할 가치가 있다.
- 다만 1일 대회 직전에는 구현/검증 비용이 커서 우선순위는 낮다.

출처:
- ACL Anthology, *UHH-LT at SemEval-2020 Task 12*  
  https://aclanthology.org/2020.semeval-1.213/
- GitHub, `uhh-lt/uhhlt-offenseval2020`  
  https://github.com/uhh-lt/uhhlt-offenseval2020

### 10.2 Jigsaw / Unitary multilingual toxic model — 다중 과제 학습과 앙상블 지향

Unitary의 multilingual toxic XLM-RoBERTa 모델 카드는 이 모델이 **여러 Jigsaw toxic comment challenge 데이터**를 바탕으로 학습되었음을 설명한다. 독성 분류 계열 대회에서는 단일 모델보다 **다중 데이터/다중 단계 학습/앙상블**이 자주 쓰인다.

시사점:
- DKTC에서도 단일 최고점 모델 1개만 들고 가는 것보다
  - epoch 성향이 다른 모델
  - lr 성향이 다른 모델
  - cosine/linear 같은 스케줄이 다른 모델
  을 여러 개 보유하는 것이 실전적으로 유리하다.
- 현재 `focal_epoch_5`, `focal_lr_5e-5`, `focal_cosine`을 동시에 보존하는 판단은 이 방향과 맞다.

출처:
- Hugging Face model card, `unitary/multilingual-toxic-xlm-roberta`  
  https://huggingface.co/unitary/multilingual-toxic-xlm-roberta

### 10.3 KOLD — 한국어 offensive language에서 macro-F1의 중요성

KOLD 논문은 한국어 offensive language dataset을 제시하며, offensive 여부 외에 target-related multi-level 분류를 다룬다. 논문에는 하위 분류 과제에서 **macro-F1**을 사용하며, 라벨 구조가 세분화될수록 성능이 크게 어려워짐을 보여준다.

시사점:
- DKTC도 단순 binary detection보다 세분화된 5-class 과제라서 macro-F1이 중요하다.
- 현재 일반대화 F1이 매우 높다고 안심하면 안 되고, `기타괴롭힘` 같은 약한 클래스가 전체 경쟁력을 깎는다.
- 즉 지금처럼 macro-F1 중심으로 의사결정하는 방향은 맞다.

출처:
- EMNLP 2022, *KOLD: Korean Offensive Language Dataset*  
  https://aclanthology.org/2022.emnlp-main.744.pdf

### 10.4 한국어 offensive/hate corpus 확장 사례 — 라벨 품질과 노이즈 완화

2023년 한국어 offensive corpus 관련 연구에서는 대규모 데이터셋 구축과 함께 **주석 노이즈와 편향 문제**, **라벨 품질 개선 필요성**을 강조한다.

시사점:
- 현재 DKTC 성능 병목이 `기타괴롭힘`처럼 경계가 넓은 클래스에 몰린 것은 모델만의 문제가 아닐 수 있다.
- 즉 추가 개선은 하이퍼파라미터보다 **라벨 경계 정의/샘플 품질/규칙 후처리**가 더 중요할 수 있다.

출처:
- Findings of EMNLP 2023, *A Hate Speech Detection Corpus in Korean with Target-Specific Ratings*  
  https://aclanthology.org/2023.findings-emnlp.952.pdf

---

## 11. 다음 세션 시작 체크리스트

다음 세션에서 바로 시작할 수 있도록 최소 체크리스트를 남긴다.

### 필수 확인
- [ ] `train_processed_260317_n_1000.csv` / `val_processed_260317_n_1000.csv` 존재 여부
- [ ] `focal_epoch_5`, `focal_lr_5e-5`, `focal_cosine` 결과와 설정 메모
- [ ] 현재 best 기준 confusion pattern 재확인
- [ ] Kaggle 대회 평가 지표 확인
- [ ] public/private leaderboard 구조 확인

### 바로 수행할 실험
- [ ] 규칙 기반 logit adjustment 1차 버전
- [ ] 2단계 분류 프로토타입
- [ ] 3-fold stratified validation 뼈대 작성

### 보류할 실험
- [ ] gamma 세밀 튜닝 반복
- [ ] class weight 미세조정 반복
- [ ] 일반대화 대량 증량

---

## 12. 최종 결론

이번 세션에서 확인된 핵심은 다음과 같다.

1. 현재 데이터와 `klue/roberta-base` 조합은 이미 강한 수준의 베이스라인을 형성했다.
2. baseline 대비 최종 최고점은 **Macro F1 +0.0154** 개선되었다.
3. Focal Loss 실험 공간에서 최고점은 `focal_epoch_5`였고, 전체 최고점은 **0.9301 macro F1**이다.
4. 하지만 진짜 병목은 여전히 **괴롭힘 내부 경계**, 특히 `기타괴롭힘`이다.
5. 다음 단계의 핵심은 더 많은 하이퍼파라미터 탐색이 아니라:
   - 최고 성능 후보 3개 유지
   - 규칙 기반 후처리
   - 2단계 분류
   - CV 안정화
   - 필요 시 도메인 적응/앙상블
   로 정리된다.

이 문서를 기준점으로 삼으면, 다음 세션에서는 같은 실험을 반복하지 않고 **현재 병목에 직접 타격하는 개선**부터 이어갈 수 있다.
