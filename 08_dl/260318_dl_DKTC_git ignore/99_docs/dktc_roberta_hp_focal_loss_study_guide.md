# DKTC 위협 대화 분류 — klue/roberta-base HP 튜닝 + Focal Loss 상세 학습자료

> **노트북**: `roberta-base_hp_Focal_Loss_v2_fin.ipynb`
> **목적**: OFAT(One-Factor-At-a-Time) 방식으로 하이퍼파라미터를 탐색하고, Focal Loss를 적용하여 5-class 위협 대화 분류 성능을 최적화한 뒤, K-Fold 앙상블로 안정성을 검증하는 전체 실험 파이프라인

---

## 전체 논리 구조 흐름도

```
[00] 환경 설정 + GPU 최적화
  ↓
[01] 데이터 로드 + [턴] 특수 토큰 등록 + 토크나이징 + 클래스 가중치 계산
  ↓
[02] 손실 함수 정의 (WeightedCE / FocalLoss / vanilla CE) + 커스텀 Trainer 클래스
  ↓
[03] OFAT 실험 목록 정의 (Baseline + 12개 변형 = 총 13개)
  ↓
[04] 실험 루프 실행 → 결과 수집
  ↓
[05] 전체 결과 비교 (Baseline 대비 delta F1)
  ↓
[06] 시각화 (바 차트 + 학습 곡선)
  ↓
[07] Best 모델 상세 분석 (classification_report + 오분류 패턴)
  ↓
[08] 결과 저장 (CSV + JSON)
  ↓
[09] 오분류 원문 + top-2 확률/마진 심층 분석
  ↓
[10] K-Fold 앙상블 (OFAT Best HP → 5-Fold OOF)
  ↓
[11] 증강 ablation 실험 가이드 (참고용)
```

**핵심 설계 원칙**: 한 번에 하나의 파라미터만 변경(OFAT) → Best HP 확정 → K-Fold로 분산 검증

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

## 부록: 전체 파라미터 탐색 범위 요약

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

## 부록: 핵심 개념 비교표

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
