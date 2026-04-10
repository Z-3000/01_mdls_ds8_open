# DKTC 위협 대화 분류 — 전체 파이프라인 상세 학습자료

> **노트북 1**: `roberta-base_hp_Focal_Loss_v2_fin.ipynb` — 학습/실험/분석
> **노트북 2**: `submission_v3.ipynb` — 추론/제출 생성
> **전체 목적**: OFAT 방식 HP 탐색 → Focal Loss 적용 → K-Fold 앙상블 검증 → 테스트셋 추론 및 제출 파일 생성까지의 end-to-end 파이프라인

---

## 전체 논리 구조 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│  노트북 1: 학습/실험/분석 (roberta-base_hp_Focal_Loss_v2)   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [00] 환경 설정 + GPU 최적화                                │
│    ↓                                                        │
│  [01] 데이터 로드 + [턴] 특수 토큰 등록 + 토크나이징         │
│       + 클래스 가중치 계산                                   │
│    ↓                                                        │
│  [02] 손실 함수 정의 (WeightedCE / FocalLoss / vanilla CE)  │
│       + 커스텀 Trainer 클래스                                │
│    ↓                                                        │
│  [03] OFAT 실험 목록 정의 (Baseline + 12개 변형 = 총 13개)  │
│    ↓                                                        │
│  [04] 실험 루프 실행 → 결과 수집                             │
│    ↓                                                        │
│  [05] 전체 결과 비교 (Baseline 대비 delta F1)               │
│    ↓                                                        │
│  [06] 시각화 (바 차트 + 학습 곡선)                           │
│    ↓                                                        │
│  [07] Best 모델 상세 분석 (classification_report + 오분류)   │
│    ↓                                                        │
│  [08] 결과 저장 (CSV + JSON)                                │
│    ↓                                                        │
│  [09] 오분류 원문 + top-2 확률/마진 심층 분석                │
│    ↓                                                        │
│  [10] K-Fold 앙상블 (OFAT Best HP → 5-Fold OOF)            │
│       → 5개 체크포인트 생성                                  │
│    ↓                                                        │
│  [11] 증강 ablation 실험 가이드 (참고용)                     │
│                                                             │
│  ★ 출력물: results_dktc_kfold_{1~5}of5/checkpoint-*        │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ 체크포인트 폴더 경로 전달
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  노트북 2: 추론/제출 생성 (submission_v3)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [S0] 공통 설정 (CONFIG, 레이블, 시드)                      │
│    ↓                                                        │
│  [S1] 학습과 동일한 텍스트 전처리 재현                       │
│       (clean_text → normalize_conversation)                 │
│    ↓                                                        │
│  [S2] 입력 검증 (test.csv ↔ submission.csv 정합성)          │
│    ↓                                                        │
│  [S3] 모델/체크포인트 로드 + [턴] 토큰 검증                 │
│    ↓                                                        │
│  [S4] 추론 (배치 단위 logit 생성)                            │
│    ↓                                                        │
│  [S5] 제출 생성 — 3가지 모드:                                │
│       ├─ 단일 모델 제출                                     │
│       ├─ 다중 단일 모델 제출                                │
│       └─ 앙상블 제출 (가중 평균 logit) ← 권장               │
│                                                             │
│  ★ 출력물: ./submission/sub_kfold_5fold_avglogits_v3.csv   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**핵심 설계 원칙**: OFAT로 HP 확정 → K-Fold로 분산 검증 + 5개 체크포인트 생성 → 5-Fold 평균 logit 앙상블로 안정적 제출

---

# Part 1: 학습/실험/분석 (roberta-base_hp_Focal_Loss_v2_fin.ipynb)

---

## 00. 환경 설정 및 최적화

### 목적
재현성 확보, GPU 활용 극대화, CPU 부하 조절

### 핵심 설정

| 설정 항목 | 값 | 근거 |
|-----------|-----|------|
| `SEED` | 42 | 재현성 확보. numpy, torch CPU/GPU 전부 고정 |
| `CPU_THREADS` | `CPU_COUNT // 2` | 전체 코어의 절반만 사용하여 시스템 부하 분산 |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `CPU_THREADS` | OpenMP/MKL 연산 병렬 스레드 수 제한 |
| `TOKENIZERS_PARALLELISM` | `"true"` | HuggingFace 토크나이저 병렬 처리 활성화 |

### GPU 가속 설정 (CUDA 사용 시)

| 설정 | 효과 |
|------|------|
| `torch.set_float32_matmul_precision("high")` | FP32 행렬 연산 정밀도를 낮춰 속도 향상 |
| `torch.backends.cuda.matmul.allow_tf32 = True` | Ampere 이상 GPU에서 TF32 가속 허용 |
| `torch.backends.cudnn.allow_tf32 = True` | cuDNN 라이브러리도 TF32 사용 |
| `torch.backends.cudnn.benchmark = True` | 입력 크기에 최적화된 연산 알고리즘 자동 탐색 |

> **TF32란**: FP32와 동일한 범위(8-bit exponent)를 가지되 mantissa가 10-bit로 줄어든 형식. 정밀도 손실은 미미하지만 Tensor Core 활용으로 속도가 크게 향상됨.

---

## 01. 데이터 로드 + [턴] 특수 토큰 등록 + 토크나이징

### 모델/데이터 기본 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `MODEL_NAME` | `klue/roberta-base` | 한국어 사전학습 RoBERTa (KLUE 벤치마크 기반) |
| `MAX_LENGTH` | 512 | 전처리 EDA에서 97% 커버리지 확인된 값 |
| `NUM_LABELS` | 5 | 협박(0), 갈취(1), 직장내괴롭힘(2), 기타괴롭힘(3), 일반대화(4) |

### [턴] 특수 토큰 등록 — 왜 필요한가

전처리 파이프라인(`normalize_conversation()`)에서 대화 턴 구분자로 `[턴]`을 삽입하였으나, 기본 토크나이저는 이를 `[`, `턴`, `]` 세 개의 서브토큰으로 분리함.

**문제**: 턴 경계라는 구조적 의미가 세 개 토큰으로 흩어져 모델이 턴 전환을 하나의 단위로 인식하지 못함.

**해결**: `add_special_tokens({"additional_special_tokens": ["[턴]"]})`으로 단일 토큰 등록

```python
TURN_TOKEN = "[턴]"
num_added = tokenizer.add_special_tokens({"additional_special_tokens": [TURN_TOKEN]})
# → vocab size가 1 증가

# 검증
tokenizer.tokenize("안녕하세요 [턴] 네 반갑습니다")
# → ['안녕하세요', '[턴]', '네', '반갑', '##습니다']  ← [턴]이 단일 토큰
```

**후속 조치** (필수): 모델의 임베딩 레이어 크기도 맞춰야 함 → `model.resize_token_embeddings(len(tokenizer))`를 매 실험 초기화 시 호출

> **학습↔추론 일관성 주의**: 학습 시 [턴] 토큰이 등록된 토크나이저가 체크포인트에 저장됨. 추론(submission_v3)에서는 체크포인트에서 토크나이저를 로드하므로 [턴] 토큰이 자동 포함됨. 단, 추론 코드에서 `TURN_TOKEN in vocab` 검증을 반드시 수행 (→ Part 2 [S3] 참조).

### 토크나이징 처리

```python
def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
```

| 설정 | 값 | 이유 |
|------|-----|------|
| `truncation=True` | MAX_LENGTH 초과 시 자름 | 512 토큰 이상 입력 방지 |
| `padding=False` | 여기서 패딩 안 함 | `DataCollatorWithPadding`이 배치 단위 동적 패딩 수행 → 불필요한 패딩 토큰 최소화 → 연산 효율 |
| `batched=True, batch_size=1000` | 배치 단위 처리 | 1건씩 처리보다 훨씬 빠름 |
| `num_proc=max(1, min(8, CPU_THREADS))` | 멀티프로세스 | 최대 8프로세스로 병렬 토크나이징 |

### 클래스 가중치 계산

```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_LABELS),
    y=train_df["label"].values
)
```

**`balanced` 공식**: `w_i = n_samples / (n_classes × n_samples_i)`

- 소수 클래스(샘플이 적은 클래스) → 가중치 높음
- 다수 클래스(샘플이 많은 클래스) → 가중치 낮음
- 이 텐서는 이후 `WeightedTrainer`와 `FocalLossTrainer`의 `alpha` 인자로 전달됨

---

## 02. 손실 함수 + Trainer 클래스 정의

### 평가 지표 함수

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    ...,
        "f1_macro":    ...,    # ← Best 모델 선정 기준
        "f1_weighted": ...,
    }
```

**F1 macro vs F1 weighted 선택 근거**:
- **F1 macro**: 모든 클래스에 동일 비중 → 소수 클래스 성능이 직접 반영됨
- **F1 weighted**: 클래스 빈도 비례 가중 → 다수 클래스에 끌림
- 위협 대화 분류에서는 소수 위협 클래스 탐지가 핵심이므로 **F1 macro가 적합**

### 손실 함수 3종 비교

#### 1) Vanilla CrossEntropy (loss_type=`"ce"`)
- HuggingFace `Trainer` 기본 내장 CE 사용
- 클래스 불균형 보정 없음
- `label_smoothing_factor` 적용 가능 (TrainingArguments에서 설정)

#### 2) Weighted CrossEntropy (loss_type=`"weighted_ce"`)

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

**동작**: CE의 `weight` 인자에 클래스별 가중치를 전달 → 소수 클래스 오분류 시 더 큰 페널티

#### 3) Focal Loss (loss_type=`"focal"`) — 이 노트북의 핵심 추가 사항

```python
class FocalLoss(torch.nn.Module):
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")  # ①
        pt = torch.exp(-ce)                                      # ②
        focal_term = (1.0 - pt) ** self.gamma * ce               # ③
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, labels)               # ④
            focal_term = alpha_t * focal_term
        return focal_term.mean()                                  # ⑤
```

**단계별 수학적 흐름**:

| 단계 | 코드 | 수학 | 의미 |
|------|------|------|------|
| ① | `F.cross_entropy(..., reduction="none")` | `CE_i = -log(p_t)` | 샘플별 CE 계산 (평균 내지 않음) |
| ② | `torch.exp(-ce)` | `p_t = exp(-CE_i) = exp(log(p_t)) = p_t` | 정답 클래스 확률 복원 |
| ③ | `(1.0 - pt) ** gamma * ce` | `(1-p_t)^γ × CE_i` | **Focal Modulation**: p_t가 높으면(쉬운 샘플) `(1-p_t)^γ ≈ 0` → 기여 감소 |
| ④ | `alpha.gather(0, labels)` | `α_t × focal_term` | 클래스별 가중치 적용 |
| ⑤ | `.mean()` | 배치 평균 | 최종 손실값 |

**gamma(γ) 값의 효과**:

| gamma | 효과 | 특성 |
|-------|------|------|
| 0 | `(1-p_t)^0 = 1` → 일반 CE와 동일 | Focal 효과 없음 |
| 1.0 | 쉬운 샘플 감쇠 약함 | 부드러운 조정 |
| 2.0 | 기본값. 적절한 감쇠 | 논문 권장값 |
| 3.0 | 감쇠 강함 | 어려운 샘플에 극도로 집중, 과적합 위험 |

**alpha를 CE의 weight 인자에 넣지 않는 이유** (코드 주석에 명시):
- CE의 `weight`에 넣으면 `ce_i = -w × log(p_t)` → `p_t = exp(-ce_i) = exp(w × log(p_t)) ≠ p_t`
- 즉, p_t 복원이 수학적으로 틀려짐
- 따라서 CE는 weight 없이 계산하고, alpha는 focal_term에 별도 곱셈

### Trainer 선택 로직

```
loss_type == "focal"       → FocalLossTrainer (gamma + alpha 주입)
loss_type == "weighted_ce" → WeightedTrainer (class_weights_tensor 사용)
loss_type == "ce"          → 기본 Trainer (label_smoothing 적용 가능)
```

---

## 03. OFAT 실험 목록 설계

### Baseline 설정

```python
BASE = dict(
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=16,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    label_smoothing_factor=0.0,
    loss_type="weighted_ce",
    focal_gamma=2.0,          # focal이 아니면 무시됨
)
```

### Phase 1: 기존 HP OFAT (10개 실험)

| # | 실험명 | 변경 파라미터 | 변경 값 | 나머지 |
|---|--------|-------------|---------|--------|
| 1 | `baseline` | - | - | BASE 그대로 |
| 2 | `lr_1e-5` | learning_rate | 1e-5 | BASE |
| 3 | `lr_5e-5` | learning_rate | 5e-5 | BASE |
| 4 | `epoch_5` | num_epochs | 5 | BASE |
| 5 | `wd_0` | weight_decay | 0.0 | BASE |
| 6 | `wd_0.1` | weight_decay | 0.1 | BASE |
| 7 | `cosine` | lr_scheduler_type | cosine | BASE |
| 8 | `no_warmup` | warmup_ratio | 0.0 | BASE |
| 9 | `smooth_0.1` | label_smoothing_factor + loss_type | 0.1 + `"ce"` | BASE |
| 10 | `no_weight` | loss_type | `"ce"` | BASE |

**주의사항 — 실험 9번**:
- `label_smoothing_factor`는 HuggingFace `Trainer` 내장 CE에서만 동작
- 커스텀 `compute_loss`를 오버라이드하는 `WeightedTrainer`/`FocalLossTrainer`에서는 무시됨
- 따라서 `loss_type="ce"`로 전환 필수

### Phase 2: Focal Loss OFAT (3개 실험)

| # | 실험명 | gamma | alpha |
|---|--------|-------|-------|
| 11 | `focal_g1.0` | 1.0 | class_weights_tensor |
| 12 | `focal_g2.0` | 2.0 | class_weights_tensor |
| 13 | `focal_g3.0` | 3.0 | class_weights_tensor |

**설계 의도**: Weighted CE(Baseline)와 Focal Loss의 성능 차이를 직접 비교. alpha는 동일한 class_weights_tensor 사용.

---

## run_experiment() 함수 상세

모든 실험의 핵심 실행 단위. OFAT 및 K-Fold 양쪽 모두에서 호출됨.

### 매 실험마다 수행하는 초기화

1. **모델 재생성**: `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)` → 이전 실험의 학습 결과가 잔류하지 않도록 매번 clean state
2. **임베딩 크기 조정**: `model.resize_token_embeddings(len(tokenizer))` → [턴] 토큰 반영
3. **BF16 지원 확인**: `torch.cuda.is_bf16_supported()` → RTX 5060 Ti는 Blackwell 아키텍처로 BF16 지원

### TrainingArguments 핵심 파라미터

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| `per_device_train_batch_size` | 16 (GPU) / 8 (CPU) | RTX 5060 Ti 16GB VRAM 기준 |
| `per_device_eval_batch_size` | 32 (GPU) / 8 (CPU) | 평가 시 gradient 불필요 → 배치 2배 가능 |
| `eval_strategy` | `"epoch"` | 매 에폭 끝에 검증 |
| `load_best_model_at_end` | `True` | 학습 종료 후 가장 좋은 체크포인트 자동 로드 |
| `metric_for_best_model` | `"f1_macro"` | Best 판단 기준 |
| `fp16` / `bf16` | BF16 우선, 미지원 시 FP16 | 혼합 정밀도 학습으로 속도↑ 메모리↓ |
| `optim` | `"adamw_torch_fused"` (GPU) | fused 옵티마이저: 파라미터 업데이트를 단일 CUDA 커널로 수행 → 속도 향상 |
| `save_total_limit` | 1 | 디스크 절약: 최고 모델 1개만 보존 |
| `dataloader_pin_memory` | `True` (GPU) | CPU→GPU 메모리 전송 시 pinned memory 사용 → 전송 속도 향상 |
| `dataloader_persistent_workers` | `True` | 에폭마다 워커 프로세스 재생성 방지 |

### 체크포인트 저장 구조 — 추론 노트북과의 연결점

```
results_dktc_{exp_name}/
  └── checkpoint-{step}/
        ├── config.json          ← 모델 아키텍처 설정
        ├── model.safetensors    ← 학습된 가중치
        ├── tokenizer.json       ← [턴] 토큰 포함된 토크나이저
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── training_args.bin
```

- `save_total_limit=1` → 각 실험 폴더에 **best 체크포인트 1개만** 존재
- `load_best_model_at_end=True` → 저장된 체크포인트가 곧 best 모델
- K-Fold 실행 시 `results_dktc_kfold_{1~5}of5/checkpoint-*` 5개 생성 → **이것이 submission_v3의 입력**

### 실험 후 메모리 정리

```python
del model, trainer
gc.collect()
torch.cuda.empty_cache()
```

13개 실험을 순차 실행하므로, 매 실험 후 GPU 캐시를 비우지 않으면 OOM 발생 가능

### 반환값 5종

| 반환값 | 용도 |
|--------|------|
| `result` (dict) | 실험 요약 (HP + 성능 지표 + 시간) |
| `epoch_logs` (list) | 에폭별 eval_loss, accuracy, f1 → 학습 곡선 시각화 |
| `val_preds` (ndarray) | 검증셋 예측 레이블 → 오분류 분석 |
| `val_labels` (ndarray) | 검증셋 정답 레이블 |
| `val_logits` (ndarray) | 검증셋 raw logits → top-2 확률/마진 분석, K-Fold 앙상블 |

---

## 04–05. 실험 루프 실행 및 결과 비교

### 실험 루프 구조

```python
for exp in EXPERIMENTS:
    result, epoch_logs, val_preds, val_labels, val_logits = run_experiment(**exp)
    all_results.append(result)
    all_epoch_logs[exp["exp_name"]] = epoch_logs
    all_val_preds[exp["exp_name"]] = val_preds
    all_val_logits[exp["exp_name"]] = val_logits
```

- 13개 실험 순차 실행
- 예측값/logits를 실험명 key로 딕셔너리에 저장 → 후속 분석에서 특정 실험 결과 즉시 접근 가능

### 결과 비교 핵심

```python
df["delta_val_f1_macro"] = df["val_f1_macro"] - baseline_row["val_f1_macro"]
```

- Baseline 대비 F1 macro 변화량을 계산하여 어떤 HP 변경이 실질적 효과가 있었는지 정량화
- `val_f1_macro` 내림차순 정렬 → Best 실험이 최상단

---

## 06. 시각화

### (a) 실험별 Val F1 Macro 바 차트
- 색상 코딩: baseline=빨강, focal 계열=초록, 나머지=파랑
- 내림차순 정렬로 시각적 순위 파악 용이

### (b) 상위 5개 실험 학습 곡선
- 에폭별 `eval_f1_macro` 추이
- Best 실험은 `linewidth=2.5`로 강조
- 과적합 여부 확인: 후반 에폭에서 F1이 하락하면 과적합 의심

---

## 07. Best 모델 상세 분석

### classification_report 출력 항목

| 지표 | 의미 | 주의점 |
|------|------|--------|
| precision | 예측한 것 중 실제 정답 비율 | 높으면 오탐(False Positive) 적음 |
| recall | 실제 정답 중 모델이 찾아낸 비율 | 높으면 미탐(False Negative) 적음 |
| f1-score | precision과 recall의 조화평균 | 둘 중 하나가 낮으면 크게 떨어짐 |
| support | 해당 클래스의 실제 샘플 수 | 적은 클래스의 지표 신뢰도 낮음 |

### 오분류 패턴 분석

```python
wrong_pairs = list(zip(all_val_labels[wrong_mask], best_preds[wrong_mask]))
Counter(wrong_pairs).most_common(10)
```

- `(실제 클래스, 예측 클래스)` 쌍을 집계
- 빈출 오분류 패턴 → 클래스 간 경계가 모호한 영역 식별
- 예: "기타괴롭힘 → 일반대화" 다발 → 기타괴롭힘의 표현이 일상 대화와 유사

---

## 08. 오분류 원문 + top-2 확률/마진 심층 분석

### top-2 분석의 의미

모델이 틀린 이유를 "단순 오분류"를 넘어 **확신도**로 구분:

| 유형 | margin 값 | 해석 |
|------|-----------|------|
| 애매한 오분류 | margin ≈ 0 (top1 ≈ top2) | 두 클래스 경계에서 헷갈림 → 데이터/레이블 문제 가능성 |
| 확신 오분류 | margin >> 0 | 모델이 강하게 확신하며 틀림 → 학습 자체의 편향 또는 노이즈 레이블 |

### softmax + margin 계산 로직

```python
def softmax_np(x):
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)           # 수치 안정성: 최대값을 빼서 overflow 방지
    return e / np.sum(e, axis=1, keepdims=True)
```

```python
margin = top1_prob - top2_prob       # 1등과 2등 확률 차이
```

### 출력물 (CSV 파일)

| 파일 | 내용 |
|------|------|
| `dktc_all_misclassified_with_top2.csv` | 전체 오분류 샘플 + top-2 확률 + 마진 |
| `dktc_misclassification_pair_summary.csv` | 오분류 패턴별 건수, 평균/최소/최대 마진 |
| `misclassified_pairs_with_top2/` 디렉토리 | 패턴별 개별 CSV (마진 오름차순 정렬) |
| `dktc_misclassified_most_ambiguous_top30.csv` | 마진이 가장 작은(가장 애매한) 오분류 30건 |
| `dktc_misclassified_most_confident_top30.csv` | 마진이 가장 큰(가장 확신하며 틀린) 오분류 30건 |

### margin 분포 시각화
- **(a) 정분류 vs 오분류 히스토그램**: 오분류의 마진 분포가 왼쪽(낮은 쪽)에 집중되어 있으면, 모델이 "잘 모르는 상태에서 틀린 것"이므로 개선 여지가 큼
- **(b) 오분류 패턴별 마진 boxplot**: 특정 패턴의 마진이 높다면 해당 패턴은 데이터 품질(레이블 오류) 점검 필요

---

## 09. K-Fold 앙상블

### 목적
- 단일 train/val split의 **분산(variance)** 을 줄여 성능 추정치의 신뢰도 확보
- Fold별 logit 평균으로 더 안정적인 예측 생성
- **5개 체크포인트를 생성하여 추론 시 앙상블 제출의 재료로 활용**

### 실행 절차

```
1. OFAT Best HP 추출 (df_sorted.iloc[0])
2. train_df + val_df 합치기 → full_df
3. full_df 전체 토크나이징 (1회)
4. StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
5. 각 Fold:
   - fold_train_ds / fold_val_ds 분리
   - run_experiment() 호출 (Best HP 그대로)
   - OOF logits 저장: oof_logits[val_idx] = val_logits
   - 체크포인트 저장: results_dktc_kfold_{i}of5/checkpoint-*
6. OOF 전체 평가: argmax(oof_logits) → oof_preds
```

### 핵심 파라미터

| 항목 | 값 | 설명 |
|------|-----|------|
| `N_FOLDS` | 5 | 표준적 K-Fold 수 |
| `StratifiedKFold` | shuffle=True, random_state=42 | 클래스 비율 유지 + 재현성 |

### OOF(Out-of-Fold) 평가

```python
oof_logits = np.zeros((len(full_df), NUM_LABELS))
# 각 Fold의 검증 인덱스 위치에 해당 Fold 모델의 logits 저장
oof_logits[val_idx] = val_logits
```

- 모든 샘플이 **정확히 1번** 검증셋에 포함됨 → 전체 데이터에 대한 편향 없는 예측
- 최종 지표: OOF Accuracy, OOF F1 macro, OOF F1 weighted
- Fold 간 F1 macro의 **표준편차** → 모델 안정성 지표

### 알려진 한계 (코드 주석에 명시)

> 현재 `train_processed` CSV에는 **증강 데이터가 포함**되어 있음. 엄밀한 K-Fold를 위해서는 증강 전 원본만으로 Fold를 나누고, 각 Fold의 학습셋에만 증강을 적용해야 함. 본 실험에서는 이 한계를 인지하고 사용.

**문제점**: 원본 A에서 증강된 A'가 있을 때, A가 train fold에, A'가 val fold에 들어가면 **데이터 누수(leakage)** 발생 → OOF 성능이 과대 추정될 수 있음.

---

## 10. 증강 ablation 실험 가이드

### 핵심 가설

> "턴 셔플(`augment_turn_shuffle`) 제거 시 F1이 올라간다면, 위협 대화의 순서 구조가 분류에 핵심적이며 셔플이 해를 끼치고 있었음."

### 실험 설계

| 실험명 | augmenters 설정 | 검증 대상 |
|--------|----------------|-----------|
| `aug_all` | turn_shuffle + random_deletion + turn_drop | 현재 기본값 (전체 증강) |
| `aug_no_shuffle` | random_deletion + turn_drop | 턴 셔플의 해악 여부 |
| `aug_deletion_only` | random_deletion만 | 최소 증강 효과 |
| `aug_none` | 증강 비활성화 | 증강 자체의 효과 유무 |

### 실행 방법
Cell 01의 `train_df = pd.read_csv(...)` 경로만 교체 → 동일 Baseline HP로 실행 → F1 비교

---

# Part 2: 추론/제출 생성 (submission_v3.ipynb)

---

## 설계 철학

이 노트북의 핵심 원칙은 **학습-추론 간 전처리 일관성 보장**과 **방어적 검증(defensive validation)**.

v3 변경사항 3가지:
1. 추론 입력을 학습과 **동일한 전처리 함수**로 고정 (clean_text → normalize_conversation)
2. 단일 모델 제출 + **다중 모델 평균 logit 앙상블** 지원
3. K-Fold 결과 폴더를 **바로 제출에 사용** 가능

> **왜 전처리 일관성이 중요한가**: 학습 시 `[턴]` 토큰 삽입, 반복문자 축소 등을 거친 텍스트로 모델이 학습됨. 추론 시 동일 전처리를 거치지 않으면, 모델이 본 적 없는 형태의 입력이 들어가 성능이 저하됨. 이것이 train-serving skew(학습-서빙 괴리)이며, 실무에서 가장 흔한 성능 하락 원인 중 하나.

---

## S0. 공통 설정 (CONFIG)

```python
CONFIG = {
    "test_path":              Path("./data/real/test.csv"),
    "sample_submission_path": Path("./data/real/submission.csv"),
    "output_dir":             Path("./submission"),
    "batch_size":             32,
    "max_length":             512,
    "seed":                   42,
    "device":                 "cuda" if torch.cuda.is_available() else "cpu",
}
```

| 파라미터 | 값 | 학습 노트북과의 관계 |
|----------|-----|---------------------|
| `max_length` | 512 | 학습과 동일 (97% 커버리지) |
| `batch_size` | 32 | 추론 전용이므로 학습(16)보다 큰 배치 가능 (gradient 불필요) |
| `seed` | 42 | 재현성 (학습과 동일) |

### 레이블 매핑

```python
LABEL_NAMES = ["협박", "갈취", "직장내괴롭힘", "기타괴롭힘", "일반대화"]
NUM_LABELS = 5
TURN_TOKEN = "[턴]"
```

**학습 노트북과 동일해야 함**. 레이블 순서가 다르면 제출 결과가 완전히 틀어짐.

---

## S1. 학습과 동일한 텍스트 전처리 재현

### 전처리 파이프라인: `clean_text()` → `normalize_conversation()`

```
test.csv의 conversation 원문
  ↓ clean_text()
정제된 텍스트
  ↓ normalize_conversation()
[턴] 토큰으로 구분된 최종 입력
```

### clean_text() 상세

```python
def clean_text(text: str) -> str:
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                  # ① 반복문자 축소
    text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1{2,}", r"\1\1", text)        # ② 자모 반복 축소
    text = re.sub(r"[^\w\s가-힣a-zA-Z0-9.,!?~\n]", " ", text)   # ③ 특수문자 제거
    text = re.sub(r"[ \t]+", " ", text)                          # ④ 연속 공백 축소
    text = re.sub(r"\n+", "\n", text)                           # ⑤ 연속 줄바꿈 축소
    return text.strip()
```

| 단계 | 정규식 | 입력 예시 | 출력 예시 | 목적 |
|------|--------|-----------|-----------|------|
| ① | `(.)\1{2,}` → `\1\1` | `ㅋㅋㅋㅋㅋ` | `ㅋㅋ` | 3회 이상 반복을 2회로 축소 |
| ② | `([ㄱ-ㅎㅏ-ㅣ])\1{2,}` → `\1\1` | `ㅎㅎㅎㅎ` | `ㅎㅎ` | 자모 반복 별도 처리 |
| ③ | `[^\w\s가-힣...]` → ` ` | `죽여버릴거야😡` | `죽여버릴거야 ` | 이모지/특수기호 제거 |
| ④ | `[ \t]+` → ` ` | `안녕     하세요` | `안녕 하세요` | 공백 정규화 |
| ⑤ | `\n+` → `\n` | 줄바꿈 3개 | 줄바꿈 1개 | 턴 구분 준비 |

### normalize_conversation() 상세

```python
def normalize_conversation(text: str) -> str:
    turns = [t.strip() for t in text.split("\n") if t.strip()]
    return f" {TURN_TOKEN} ".join(turns)
```

**동작**: 줄바꿈(`\n`)으로 분리된 각 대화 턴을 `[턴]` 토큰으로 연결

**예시**:
```
입력: "죽여버릴거야\n뭐라고?\n잘못했습니다"
출력: "죽여버릴거야 [턴] 뭐라고? [턴] 잘못했습니다"
```

### build_inference_texts() — 빈 문자열 방어

```python
def build_inference_texts(conversation_series):
    conversation_clean = conversation_series.apply(clean_text)
    conversation_norm  = conversation_clean.apply(normalize_conversation)

    empty_mask = conversation_norm.str.len().eq(0)
    if empty_mask.any():
        raise ValueError(f"전처리 후 빈 문자열이 발생했습니다: {int(empty_mask.sum())}개")

    return conversation_norm.tolist()
```

**방어 로직**: 전처리 결과가 빈 문자열이면 즉시 예외 발생. 빈 입력이 모델에 들어가면 예측이 무의미해지기 때문.

---

## S2. 입력 검증 — 4단계 방어적 검증

| 검증 단계 | 함수 | 검증 내용 | 실패 시 |
|-----------|------|-----------|---------|
| ① 컬럼 존재 | `validate_inputs()` | test.csv에 `conversation` 컬럼 존재 여부 | ValueError |
| ② 결측치 | `validate_inputs()` | conversation 컬럼에 NaN 존재 여부 | ValueError (개수 표시) |
| ③ 행 수 일치 | `validate_inputs()` | test.csv 행 수 == submission.csv 행 수 | ValueError |
| ④ ID 정렬 | `validate_id_alignment()` | test.csv의 idx와 submission.csv의 id_col 값이 동일 순서인지 | ValueError (불일치 개수 표시) |

### submission 컬럼 자동 탐지

```python
def detect_submission_columns(sample_df):
    id_candidates     = ["file_name", "idx", "id"]
    target_candidates = ["class", "target", "label"]
    # 순서대로 탐색하여 첫 매칭 사용
```

**설계 의도**: 대회/과제마다 submission 파일의 컬럼명이 다를 수 있음 (class, target, label 등). 하드코딩 대신 후보군에서 자동 탐지하여 범용성 확보.

---

## S3. 모델/체크포인트 로드

### 체크포인트 경로 자동 해결 — `resolve_model_dir()`

```python
def resolve_model_dir(model_path):
    path = Path(model_path)

    # Case 1: 이미 checkpoint 폴더 (config.json 존재)
    if (path / "config.json").exists():
        return path

    # Case 2: 상위 실험 폴더 → 내부 checkpoint-* 자동 탐색
    ckpt_dirs = sorted(
        [p for p in path.glob("checkpoint-*") if (p / "config.json").exists()],
        key=_checkpoint_step    # checkpoint-150 → 150으로 정렬
    )
    return ckpt_dirs[-1]        # 가장 높은 step의 checkpoint 선택
```

**두 가지 입력 방식 지원**:

| 입력 | 예시 | 동작 |
|------|------|------|
| checkpoint 폴더 직접 지정 | `./results_dktc_kfold_1of5/checkpoint-450` | 그대로 사용 |
| 상위 실험 폴더 지정 | `./results_dktc_kfold_1of5` | 내부 `checkpoint-*` 중 가장 높은 step 자동 선택 |

**step 기준 정렬 이유**: `save_total_limit=1`이지만, 혹시 여러 체크포인트가 남아있을 경우 가장 최신(가장 많이 학습된) 것을 선택

### 모델 로드 + 3단계 검증 — `load_model_bundle()`

```python
def load_model_bundle(model_path, device):
    resolved_path = resolve_model_dir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(resolved_path))

    # 검증 1: num_labels 일치 확인
    assert model.config.num_labels == NUM_LABELS  # 5

    # 검증 2: pad_token 존재 확인 (없으면 eos_token으로 대체)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 검증 3: [턴] 특수 토큰이 토크나이저에 등록되어 있는지 확인
    assert TURN_TOKEN in tokenizer.get_added_vocab() or TURN_TOKEN in tokenizer.get_vocab()

    model.to(device)
    model.eval()                # 드롭아웃 비활성화, BatchNorm 고정
    return tokenizer, model, resolved_path
```

| 검증 항목 | 확인 대상 | 실패 원인 |
|-----------|-----------|-----------|
| `num_labels == 5` | 모델 출력 차원 | 다른 과제의 체크포인트를 잘못 지정 |
| `pad_token` 존재 | 토크나이저 패딩 처리 | 일부 모델(GPT 계열)은 pad_token 미설정 |
| `[턴]` 토큰 존재 | 학습 시 등록된 특수 토큰 | 저장 안 된 토크나이저 사용 또는 잘못된 체크포인트 |

---

## S4. 추론 — `predict_logits()`

```python
@torch.no_grad()
def predict_logits(texts, tokenizer, model, device, batch_size=32, max_length=512):
    logits_all = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            padding=True,           # ← 학습과 다른 점: 여기서 직접 패딩
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy().astype(np.float32)
        logits_all.append(logits)

    return np.vstack(logits_all)
```

### 학습 vs 추론 토크나이징 차이점

| 항목 | 학습 (노트북 1) | 추론 (노트북 2) |
|------|----------------|----------------|
| 패딩 | `padding=False` → `DataCollatorWithPadding`이 배치 내 동적 패딩 | `padding=True` → 토크나이저가 배치 내 최대 길이로 직접 패딩 |
| 이유 | Trainer가 DataCollator 사용 | 수동 배치 루프이므로 직접 처리 |
| 결과 | 동일 (배치 내 최대 길이로 패딩) | 동일 |

### 핵심 설정

| 항목 | 값 | 설명 |
|------|-----|------|
| `@torch.no_grad()` | 데코레이터 | gradient 계산 비활성화 → VRAM 절약 + 속도 향상 |
| `batch_size` | 32 | 추론은 gradient 없으므로 학습(16)보다 2배 가능 |
| `.detach().cpu().numpy()` | 체인 호출 | GPU 텐서 → CPU → numpy 변환 (후처리 위해) |
| `.astype(np.float32)` | 타입 고정 | 메모리 효율 + 수치 일관성 |

### 출력 검증

```python
if logits_all.shape != (len(texts), NUM_LABELS):
    raise ValueError(...)
```

`(샘플 수, 5)` shape가 아니면 즉시 실패 → silent error 방지

---

## S5. 제출 생성 — 3가지 모드

### 모드 1: 단일 모델 제출 — `make_submission()`

```python
make_submission(
    model_path="./results_dktc_focal_g2.0",
    output_name="sub_focal_g2.0_v3.csv"
)
```

**흐름**: 전처리 → 모델 1개 로드 → logit 생성 → argmax → CSV 저장

### 모드 2: 다중 단일 모델 제출 — `make_multiple_submissions()`

```python
single_jobs = [
    {"model_path": "./results_dktc_wd_0.1",     "output_name": "sub_wd_0.1_v3.csv"},
    {"model_path": "./results_dktc_no_weight",   "output_name": "sub_no_weight_v3.csv"},
    {"model_path": "./results_dktc_focal_g2.0",  "output_name": "sub_focal_g2.0_v3.csv"},
]
make_multiple_submissions(single_jobs)
```

**흐름**: 전처리 **1회** → 모델 N개 순차 로드/추론 → 각각 별도 CSV 저장

**효율성**: `prepare_inference_bundle()`을 루프 밖에서 1회만 호출하여 test.csv 읽기 + 전처리를 반복하지 않음

### 모드 3: 앙상블 제출 — `make_ensemble_submission()` ★ 권장

```python
kfold_model_paths = [
    "./results_dktc_kfold_1of5",
    "./results_dktc_kfold_2of5",
    "./results_dktc_kfold_3of5",
    "./results_dktc_kfold_4of5",
    "./results_dktc_kfold_5of5",
]

kfold_submit_df, kfold_logits, kfold_output_path, kfold_resolved_paths = make_ensemble_submission(
    model_paths=kfold_model_paths,
    output_name="sub_kfold_5fold_avglogits_v3.csv",
)
```

### 앙상블 logit 계산 상세

```python
# 가중치 기본값: 동일 가중치 [1.0, 1.0, 1.0, 1.0, 1.0]
weights = np.asarray(weights, dtype=np.float64)

# 가중 합산
ensemble_logits = None
for model_path, weight in zip(model_paths, weights):
    logits = predict_logits(...)                    # 각 모델의 raw logits
    if ensemble_logits is None:
        ensemble_logits = weight * logits           # 첫 모델
    else:
        ensemble_logits += weight * logits          # 누적 합산

# 가중 평균
ensemble_logits /= float(weights.sum())

# 최종 예측
preds = np.argmax(ensemble_logits, axis=1)
```

**수학적 의미**:

```
ensemble_logit_j = Σ(w_i × logit_i_j) / Σ(w_i)

여기서:
  i = 모델 인덱스 (1~5)
  j = 클래스 인덱스 (0~4)
  w_i = 모델 i의 가중치 (기본: 전부 1.0)
```

### 왜 softmax 확률이 아닌 raw logit을 평균하는가

| 방법 | 수식 | 특성 |
|------|------|------|
| **logit 평균** (본 코드) | `argmax(mean(logits))` | 모델 간 확신도 차이가 보존됨. 한 모델이 매우 확신하면 결과에 더 크게 반영 |
| softmax 확률 평균 | `argmax(mean(softmax(logits)))` | softmax가 확률을 0~1로 압축하여 확신도 차이가 완화됨 |

logit 평균이 일반적으로 더 좋은 앙상블 성능을 보임 (특히 모델 간 성능 차이가 있을 때).

### 가중치 검증 (방어 로직)

```python
if weights.shape[0] != len(model_paths):    # 길이 불일치
    raise ValueError(...)
if np.any(weights < 0):                     # 음수 가중치
    raise ValueError(...)
if float(weights.sum()) <= 0:               # 합계 0 이하
    raise ValueError(...)
```

### 커스텀 가중치 사용 예시

```python
# Fold별 Val F1을 가중치로 사용 (성능 좋은 Fold에 더 높은 비중)
make_ensemble_submission(
    model_paths=kfold_model_paths,
    output_name="sub_kfold_weighted.csv",
    weights=[0.82, 0.85, 0.79, 0.83, 0.81],  # 각 Fold의 Val F1 macro
)
```

### 메모리 관리

```python
def _cleanup_model_objects(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- 모델을 **순차적으로 로드/추론/삭제** → 5개 모델을 동시에 메모리에 올리지 않음
- 각 모델 추론 후 `gc.collect()` + `torch.cuda.empty_cache()` → OOM 방지

### 제출 파일 저장 + 검증

```python
def save_submission(sample_df, target_col, preds, output_name, config):
    submit_df = sample_df.copy()
    submit_df[target_col] = preds.astype(int)

    submit_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 저장 후 요약 출력
    print(f"rows       : {len(submit_df)}")
    print(f"label_dist : {submit_df[target_col].value_counts().sort_index().to_dict()}")
```

| 항목 | 설명 |
|------|------|
| `sample_df.copy()` | 원본 submission 템플릿의 구조(컬럼명, 행 순서)를 그대로 유지 |
| `encoding="utf-8-sig"` | Excel에서 한글 깨짐 방지 (BOM 포함) |
| `label_dist` 출력 | 클래스별 예측 분포를 확인하여 편향 여부 즉시 파악 |

### `logits_to_preds()` — 예측값 안전 변환

```python
def logits_to_preds(logits):
    preds = np.argmax(logits, axis=1).astype(int)

    # 범위 검증: 0~4 밖의 값이 있으면 즉시 예외
    if not np.isin(preds, np.arange(NUM_LABELS)).all():
        raise ValueError(f"허용 범위(0~4) 밖 라벨: {bad_values}")

    return preds
```

**방어 목적**: logits shape가 잘못되어 argmax 결과가 0~4 범위를 벗어나는 경우를 포착

---

## 두 노트북 간 데이터/설정 일관성 체크리스트

학습과 추론 사이의 불일치가 발생하면 성능이 저하되거나 결과가 무의미해짐. 아래 항목을 반드시 확인:

| 항목 | 학습 노트북 (v2) | 추론 노트북 (v3) | 불일치 시 증상 |
|------|-----------------|-----------------|---------------|
| `MAX_LENGTH` | 512 | 512 | 추론 시 truncation 지점 다름 → 예측 불일치 |
| `NUM_LABELS` | 5 | 5 | 모델 출력 차원 불일치 → 로드 에러 |
| `LABEL_NAMES` 순서 | [협박, 갈취, 직장내괴롭힘, 기타괴롭힘, 일반대화] | 동일 | 순서 다르면 레이블 뒤바뀜 |
| `TURN_TOKEN` | `[턴]` | `[턴]` | 토큰 불일치 → 턴 구분 인식 실패 |
| `clean_text()` | 전처리 파이프라인 내장 | 동일 함수 재구현 | 전처리 차이 → train-serving skew |
| `normalize_conversation()` | 전처리 파이프라인 내장 | 동일 함수 재구현 | 턴 구분 형식 불일치 |
| `SEED` | 42 | 42 | 재현성 (추론에서는 영향 미미) |

---

## 부록 A: 전체 파라미터 탐색 범위 요약

| 파라미터 | 탐색 값 | Baseline |
|----------|---------|----------|
| learning_rate | 1e-5, **2e-5**, 5e-5 | 2e-5 |
| num_epochs | **3**, 5 | 3 |
| weight_decay | 0.0, **0.01**, 0.1 | 0.01 |
| lr_scheduler_type | **linear**, cosine | linear |
| warmup_ratio | 0.0, **0.1** | 0.1 |
| label_smoothing_factor | **0.0**, 0.1 | 0.0 |
| loss_type | ce, **weighted_ce**, focal | weighted_ce |
| focal_gamma | 1.0, 2.0, 3.0 | (해당 없음) |

> **OFAT 전략의 한계**: 파라미터 간 상호작용(interaction effect)을 포착하지 못함. 예를 들어 cosine scheduler + lr 5e-5 조합이 개별적으로는 안 좋아도 함께 쓰면 좋을 수 있음. 이를 보완하려면 Best Combo 실험 또는 그리드/랜덤 서치가 필요.

---

## 부록 B: 핵심 개념 비교표

### Weighted CE vs Focal Loss

| 구분 | Weighted CE | Focal Loss |
|------|-------------|------------|
| 보정 대상 | **클래스 빈도** 불균형 | 클래스 빈도 + **샘플 난이도** |
| 수식 | `w_i × CE_i` | `α_t × (1-p_t)^γ × CE_i` |
| 쉬운 샘플 처리 | 동일하게 반영 | 기여 감쇠 (γ로 조절) |
| 어려운 샘플 처리 | 동일하게 반영 | 상대적 비중 증가 |
| 추가 HP | 없음 | gamma (감쇠 강도) |
| 적합 상황 | 단순 클래스 불균형 | 불균형 + 클래스 경계 모호 |

### F1 macro vs F1 weighted

| 구분 | F1 macro | F1 weighted |
|------|----------|-------------|
| 계산 | 클래스별 F1의 단순 평균 | 클래스별 F1의 support 가중 평균 |
| 소수 클래스 영향 | 동일 비중 (1/N) | support 비례 (작음) |
| 적합 상황 | **소수 클래스 탐지 중요** | 전체 정확도 중시 |

### 단일 모델 제출 vs 앙상블 제출

| 구분 | 단일 모델 | K-Fold 앙상블 |
|------|-----------|--------------|
| 모델 수 | 1 | 5 (Fold별 1개) |
| 추론 시간 | 1× | 5× |
| 분산 | 높음 (split 의존적) | 낮음 (5개 평균) |
| 과적합 위험 | 상대적 높음 | 완화됨 |
| 일반적 성능 | baseline | +0.5~2% F1 향상 기대 |
| 제출 방식 | `make_submission()` | `make_ensemble_submission()` |

---

## 부록 C: 전체 파일 구조 (학습 → 추론)

```
프로젝트 루트/
├── data/
│   ├── train_processed_260317_n_1000.csv     ← 학습 데이터 (증강 포함)
│   ├── val_processed_260317_n_1000.csv       ← 검증 데이터
│   └── real/
│       ├── test.csv                          ← 테스트 데이터 (레이블 없음)
│       └── submission.csv                    ← 제출 템플릿
│
├── roberta-base_hp_Focal_Loss_v2_fin.ipynb   ← 학습 노트북
├── submission_v3.ipynb                       ← 추론 노트북
│
├── results_dktc_baseline/checkpoint-*/       ← OFAT 실험 체크포인트들
├── results_dktc_lr_1e-5/checkpoint-*/
├── results_dktc_focal_g2.0/checkpoint-*/
├── ...
│
├── results_dktc_kfold_1of5/checkpoint-*/     ← K-Fold 체크포인트 (추론 입력)
├── results_dktc_kfold_2of5/checkpoint-*/
├── results_dktc_kfold_3of5/checkpoint-*/
├── results_dktc_kfold_4of5/checkpoint-*/
├── results_dktc_kfold_5of5/checkpoint-*/
│
├── submission/
│   └── sub_kfold_5fold_avglogits_v3.csv      ← 최종 제출 파일
│
├── dktc_hp_focal_v2_experiment_results.csv   ← OFAT 실험 결과
├── dktc_hp_focal_v2_epoch_logs.json          ← 에폭별 학습 로그
├── dktc_kfold_results.csv                    ← K-Fold 결과
├── dktc_kfold_oof_logits.npy                 ← OOF logits
├── dktc_all_misclassified_with_top2.csv      ← 오분류 분석
└── dktc_misclassification_pair_summary.csv   ← 오분류 패턴 요약
```
