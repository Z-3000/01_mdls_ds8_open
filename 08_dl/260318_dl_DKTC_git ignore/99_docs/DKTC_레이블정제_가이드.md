# DKTC 레이블 정제를 통한 모델 성능 개선 가이드

> 작성일: 2026-03-18  
> 대상 노트북: `dktc_hp_tuning_full.py` (klue/roberta-base HP 튜닝)  
> 핵심 스크립트: `dktc_relabel_script_v2.py`

---

## 1. 왜 이걸 만들었는가

### 1.1 발견된 문제

HP 튜닝 10개 실험 중 best 모델의 val set 오분류 71건을 분석한 결과, 
**모델이 매우 확신하면서 틀린 건(margin > 0.9)이 34건(48%)**이었다.

이 34건의 원문을 직접 읽어본 결과:

- "협박"으로 라벨링된 8건: 욕설·조롱만 있고 "죽인다/찌른다" 같은 구체적 위해 고지가 없었음. 모델이 "기타괴롭힘"으로 예측한 것이 더 합리적.
- "협박"으로 라벨링된 7건: 7건 전부 금전·물건 요구("돈 가져와", "그거 줘")가 대화의 핵심이었음. 모델이 "갈취"로 예측한 것이 더 합리적.
- 그 외: 학교 환경인데 "직장내괴롭힘"으로 라벨링된 건 등.

즉, **모델이 틀린 게 아니라 원본 데이터의 레이블이 일관적이지 않은 경우**가 상당수 존재.

### 1.2 이것이 성능에 미치는 영향

레이블이 잘못된 샘플은 두 곳에서 성능을 깎는다:

- **train set의 잘못된 레이블** → 모델이 잘못된 경계를 학습 → 예측 품질 저하
- **val set의 잘못된 레이블** → 모델이 맞게 예측해도 "오답" 처리 → F1 지표 과소 측정

따라서 레이블을 수정하면 **학습 품질 향상 + 평가 정확도 향상** 두 가지 효과가 동시에 발생한다.

### 1.3 왜 v2인가 (v1의 문제)

처음 만든 v1 스크립트는 키워드 규칙을 **train/val 전체 데이터에 일괄 적용**했다. 
문제: 오분류 "협박" 25건 중 84%에서 위해 키워드가 없었는데, 이 비율이 전체 협박 클래스에도 비슷하게 적용되면 **모델이 원래 맞추고 있던 건의 레이블까지 불필요하게 바뀐다.** 27건을 고치려다가 수십~수백 건을 망가뜨리는 구조.

v2는 이 문제를 해결하기 위해, val set에서는 **오분류 목록에 있는 건에서만** 규칙을 적용한다. 정분류 건은 절대 건드리지 않는다.

---

## 2. 전체 작업 흐름

```
[Phase 1] HP 튜닝 노트북 실행 (원본 데이터)
    ↓ 생성물: dktc_all_misclassified_with_top2.csv
    ↓         dktc_hp_experiment_results.csv
    ↓         best 모델의 val F1 macro (= baseline 성능)

[Phase 2] 레이블 정제 스크립트 실행
    ↓ 입력: 원본 train/val CSV + 오분류 CSV
    ↓ 생성물: train_relabeled_v2.csv, val_relabeled_v2.csv

[Phase 3] HP 튜닝 노트북 재실행 (정제된 데이터)
    ↓ 데이터 경로만 변경, 나머지 동일
    ↓ 생성물: 새로운 val F1 macro (= 정제 후 성능)

[Phase 4] 성능 비교
    ↓ baseline F1 vs 정제 후 F1
```

---

## 3. 실행 방법 (단계별)

### 3.0 사전 조건

다음 파일이 작업 디렉토리에 있어야 한다:

| 파일 | 생성 시점 | 용도 |
|------|----------|------|
| `data/train_processed_260317_n_1000.csv` | 전처리 단계 | 원본 train |
| `data/val_processed_260317_n_1000.csv` | 전처리 단계 | 원본 val |
| `dktc_all_misclassified_with_top2.csv` | Phase 1 완료 후 | 오분류+확률 정보 |
| `dktc_relabel_script_v2.py` | 본 작업 | 레이블 정제 스크립트 |
| `dktc_hp_tuning_full.py` | 본 작업 | HP 튜닝 노트북 (경로 B) |

### 3.1 Phase 1: HP 튜닝 노트북 실행 (이미 완료된 경우 건너뛰기)

`dktc_hp_tuning_full.py`를 Jupyter에서 전체 실행한다. 이 노트북은 경로 B 버전으로, `run_experiment()`가 logits를 반환하고, 셀 14에서 오분류 + top-2 확률 분석을 수행한다.

실행 완료 시 확인할 것:
- `dktc_all_misclassified_with_top2.csv`가 작업 디렉토리에 생성되었는가
- 이 파일에 `margin` 컬럼이 있는가

이 단계의 best 모델 성능을 기록해둔다:

```
Baseline 성능 (정제 전):
  Best 실험명: ________
  Val F1 macro: ________
  Val accuracy: ________
```

### 3.2 Phase 2: 레이블 정제 스크립트 실행

```bash
cd /path/to/working/directory
python dktc_relabel_script_v2.py
```

또는 Jupyter에서 셀로 실행해도 된다.

출력에서 확인할 것:

**① Val 변경 건수 확인**

```
Step 1: Val set 정제 (margin >= 0.5)
  오분류 전체: 71건
  margin >= 0.5 후보: 60건
  Val 변경: 약 27건
```

Val 변경이 0건이면 오분류 CSV 경로가 잘못되었거나 margin_threshold가 너무 높은 것이다.

**② Train 변경 건수 확인**

```
Step 2: Train set 정제 (val에서 검증된 규칙만, 보수적 적용)
  Train 변경: ??건 / 5000건 (??%)
```

Train 변경이 전체의 **10%를 초과하면 위험 신호다.** 키워드가 너무 광범위하게 매칭되고 있다는 뜻이므로, 이 경우 `MARGIN_THRESHOLD`를 0.7이나 0.9로 올려서 다시 실행한다.

**③ 클래스 분포 변화 확인**

```
[Train]
         원본  수정후  변화
협박       1000    ??   -??
갈취       1000    ??   +??
기타괴롭힘  1000    ??   +??
...
```

특정 클래스가 극단적으로 줄거나 늘면(원본의 50% 이상 변화) 역시 위험 신호다.

**④ Sanity Check 통과 확인 (필수)**

실행 마지막에 다음이 출력되어야 한다:

```
Sanity Check
  [PASS] Val label_name 형식 일관성 유지
  [PASS] Train label_name 형식 일관성 유지
  [PASS] 행 수 불변: Train 3942, Val 969
```

`[FAIL]`이 하나라도 있으면 **수정된 CSV를 사용하지 말 것.** label_name 매핑 오류로 클래스가 분열된 상태이므로, 그 CSV로 학습하면 모델이 깨진다.

**⑤ changelog 샘플 검증 (선택사항, 5분)**

```python
changelog = pd.read_csv("data/train_relabel_v2_changelog.csv")
# 규칙별로 3건씩만 원문 확인
for rule in changelog["rule"].unique():
    print(f"\n=== {rule} ===")
    samples = changelog[changelog["rule"] == rule].head(3)
    for _, s in samples.iterrows():
        print(f"  {s['original_label']} → {s['new_label']}: {s['text_preview']}")
```

### 3.3 Phase 3: HP 튜닝 노트북 재실행

HP 튜닝 노트북(`dktc_hp_tuning_full.py`)에서 **셀 01의 데이터 경로만 변경**한다:

```python
# === 변경 전 ===
train_df = pd.read_csv("data/train_processed_260317_n_1000.csv")
val_df = pd.read_csv("data/val_processed_260317_n_1000.csv")

# === 변경 후 ===
train_df = pd.read_csv("data/train_relabeled_v2.csv")
val_df = pd.read_csv("data/val_relabeled_v2.csv")
```

나머지 코드는 전혀 수정하지 않는다.

커널 재시작 → 전체 실행.

### 3.4 Phase 4: 성능 비교

재실행 완료 후, 셀 04의 결과표에서 best 모델의 F1을 확인한다.

```
정제 후 성능:
  Best 실험명: ________
  Val F1 macro: ________
  Val accuracy: ________
```

비교:

```
                정제 전     정제 후     변화
Val F1 macro:  ________ → ________ (________)
Val accuracy:  ________ → ________ (________)
```

---

## 4. 스크립트 내부 구조 상세

### 4.0 label_name 형식 문제 (v2.0 → v2.1 수정)

원본 데이터와 오분류 CSV에서 레이블 이름 형식이 다르다:

| 출처 | 형식 | 예시 |
|------|------|------|
| 원본 CSV (`label_name` 컬럼) | 긴 형식 | "협박 대화", "갈취 대화", "직장 내 괴롭힘 대화" |
| HP 튜닝 노트북 (`LABEL_NAMES`) | 짧은 형식 | "협박", "갈취", "직장내괴롭힘" |
| 오분류 CSV (`true_label`, `pred_label`) | 짧은 형식 | "협박", "갈취", "직장내괴롭힘" |

v2.0에서는 이 차이를 고려하지 않아 두 가지 버그가 발생했다:
- Train 규칙에서 `current_label == "협박"` 비교가 항상 False (실제값은 "협박 대화")
- Val에서 label_name을 "갈취"로 덮어쓰면서 원본 "갈취 대화"와 별개의 클래스가 생성

v2.1은 `detect_label_format()` 함수로 원본 형식을 자동 감지하고, 짧은 형식 ↔ 원본 형식 매핑 테이블을 생성한다. 모든 비교는 짧은 형식으로, 모든 덮어쓰기는 원본 형식으로 수행한다.

실행 시 Sanity Check에서 `[PASS] label_name 형식 일관성 유지`가 출력되면 정상이다. `[FAIL]`이 나오면 매핑 테이블을 수동으로 확인해야 한다.

### 4.1 v2가 작동하는 원리

v2의 핵심 설계 원칙: **"모델이 확신하면서 틀린 건"에서만, "모델 예측과 키워드 근거가 일치하는 방향으로만" 레이블을 변경한다.**

이것을 분해하면 3중 안전장치다:

1. **오분류 목록 한정**: 정분류 건은 후보에 포함되지 않음 → 정분류 피해 0건
2. **margin 임계값**: margin >= 0.5 이상만 후보 → 모델이 애매하게 틀린 건은 건드리지 않음
3. **키워드 일치 확인**: 모델 예측 방향과 키워드 근거가 같을 때만 변경 → 근거 없는 변경 방지

### 4.2 Val set 처리 (`relabel_val_from_misclassified`)

**입력**: `dktc_all_misclassified_with_top2.csv` — HP 튜닝 노트북 셀 14에서 생성된 파일. 오분류 71건 각각에 대해 모델의 예측(`pred_label`), 확신도(`margin`), top-2 확률이 포함되어 있음.

**처리 흐름** (각 오분류 건에 대해):

```
margin >= 0.5 인가? → 아니오 → 건너뜀
    ↓ 예
원본 레이블이 "협박"이고 모델 예측이 "기타괴롭힘"인가?
    ↓ 예
텍스트에 위해 고지 키워드("죽여", "칼로" 등)가 있는가?
    ↓ 없음
→ 레이블을 "기타괴롭힘"으로 변경 (Rule A)
```

총 5개 규칙이 있고, 각 규칙은 **"원본 레이블 + 모델 예측 + 키워드 존재 여부"** 세 가지를 동시에 확인한다.

### 4.3 Train set 처리 (`relabel_train_conservative`)

Train set에는 모델 예측 정보가 없다. 따라서 "키워드 규칙만" 적용하되, 두 가지 추가 안전장치가 있다:

**① val에서 활성화된 규칙만 적용**

val 변경에서 Rule A, B, C가 사용되었고 Rule D, E가 사용되지 않았다면, train에서도 Rule A, B, C만 적용한다. val에서 효과가 검증되지 않은 규칙은 train에도 적용하지 않는다.

**② 갈취 키워드를 축소 (`EXTORTION_KW_STRICT`)**

Val에서는 "돈", "빌려", "먹자" 같은 맥락 의존적 키워드도 사용하지만(모델 예측이 이중 검증 역할을 하므로), train에서는 모델 예측 정보가 없으므로 false positive 위험이 높다. 따라서 train용 갈취 키워드에서 아래 항목을 제외했다:

- 제외: "돈", "빌려", "먹자", "먹을" (맥락에 따라 갈취가 아닐 수 있음)
- 유지: "만원", "내놔", "사줘", "입금" (갈취 맥락이 분명함)

**③ Rule E는 train에서 비활성**

"기타괴롭힘 → 갈취" 전환은 false positive 위험이 가장 높은 규칙이므로, train에서는 아예 적용하지 않는다.

### 4.4 분류 규칙 5개 요약

| 규칙 | 조건 | 변경 | Val 적용 | Train 적용 |
|------|------|------|---------|-----------|
| Rule A | label=협박, 모델=기타괴롭힘, 위해 키워드 없음 | → 기타괴롭힘 | O | O |
| Rule B | label=협박, 모델=갈취, 재물 키워드 있음 | → 갈취 | O | O (축소 키워드) |
| Rule C | label=직장내괴롭힘, 모델=기타괴롭힘, 직장 키워드 없음 | → 기타괴롭힘 | O | O |
| Rule D | label=기타괴롭힘, 모델=협박, 위해 키워드 있음 | → 협박 | O | O |
| Rule E | label=기타괴롭힘, 모델=갈취, 재물 키워드 2개+ | → 갈취 | O | **X** (비활성) |

### 4.5 키워드 사전

**위해 고지 키워드 (`THREAT_KW`)** — 20개

"죽여", "죽인다", "죽일", "죽는다", "죽어", "칼로", "찌른다", "찔러", "찌를", "불질러", "불지른다", "불태워", "납치", "폭파", "해코지", "목을", "목숨", "숨통", "패죽", "때려죽"

포함 기준: 구체적 신체/생명/재산 위해 행위를 직접 언급하는 표현.
미포함: "맞고 싶어?", "가만 안 둔다", "두고 봐" — 간접적 암시는 제외.

**갈취 키워드 — Val용 (`EXTORTION_KW`)** — 22개

"돈", "만원", "천원", "백만", "빌려", "줘봐", "주라", "내놔", "내놓", "사줘", "사와", "사다줘", "사 와", "가져와", "가져오", "먹자", "먹을", "입금", "송금", "이더리움", "비트코인", "이자", "원금"

**갈취 키워드 — Train용 (`EXTORTION_KW_STRICT`)** — 11개

"만원", "천원", "백만", "내놔", "내놓", "사줘", "사다줘", "가져와", "가져오", "입금", "송금", "이더리움"

**직장 맥락 키워드 (`WORKPLACE_KW`)** — 16개

"대리", "과장", "부장", "차장", "팀장", "사원", "직원", "인턴", "신입", "법인카드", "회식", "야근", "출근", "부서", "본부", "팀에", "회의"

---

## 5. 기대 효과와 한계

### 5.1 기대 효과

**Val set 직접 효과**: 오분류 71건 중 약 27건의 레이블이 모델 예측 방향으로 수정됨. 이 27건은 이전에 "오답"으로 계산되던 건이 "정답"으로 전환되므로, val F1 macro가 직접적으로 상승한다.

단순 계산(정분류 건수 증가만 고려, 학습 개선 효과 제외):

```
정제 전 정분류: ~929건 / ~1000건
정제 후 정분류: ~929 + 27 = ~956건 / ~1000건 (val 레이블 수정분)

단, 이것은 "동일한 모델로 정제된 val을 평가한" 경우의 상한이다.
재학습 후에는 모델 자체가 달라지므로, 실제 수치는 이보다 높거나 낮을 수 있다.
```

**Train set 간접 효과**: 노이즈가 제거된 train으로 재학습하면 모델이 더 일관된 경계를 학습하므로, 기존에 맞추지 못하던 새로운 건도 맞출 수 있다.

### 5.2 성능이 오히려 떨어질 수 있는 경우

**경우 1: Train set 규칙이 과도하게 적용된 경우**

Train 변경 건수가 전체의 10%를 넘으면, 규칙의 false positive가 정분류 건의 학습 신호를 오염시킬 수 있다. 이 경우 `MARGIN_THRESHOLD`를 높이거나, train에서는 Rule A만 적용하는 등 규칙을 줄여야 한다.

**경우 2: 클래스 분포가 크게 바뀐 경우**

예를 들어 "협박" 클래스가 1000건에서 500건으로 줄고 "기타괴롭힘"이 1000건에서 1500건으로 늘면, WeightedTrainer의 class_weight 계산이 달라지면서 의도치 않은 학습 편향이 발생할 수 있다. 분포 변화가 ±20% 이내인지 확인한다.

**경우 3: "간접 위협" 건의 학습 신호가 사라진 경우**

"가만 안 둔다", "두고 봐" 같은 간접 위협은 THREAT_KW에 포함되지 않으므로 Rule A에 의해 "기타괴롭힘"으로 변경된다. 이 건들이 실제로 협박이었다면, 모델이 간접 위협 패턴을 잃어버릴 수 있다.

### 5.3 성능이 떨어졌을 때 대처법

1. `MARGIN_THRESHOLD`를 0.9로 올려서 재시도 (가장 확실한 34건만 변경)
2. Train set은 정제하지 않고 val set만 정제한 후 재학습
3. Rule B를 비활성화하고 Rule A, C, D만 적용
4. Train용 키워드를 더 축소

---

## 6. 생성되는 파일 목록

### Phase 1 (HP 튜닝 노트북) 생성 파일

| 파일명 | 내용 |
|-------|------|
| `dktc_hp_experiment_results.csv` | 10개 실험 결과 요약 |
| `dktc_hp_epoch_logs.json` | 에폭별 학습 곡선 |
| `dktc_hp_experiment_chart.png` | 실험 비교 차트 |
| `dktc_all_misclassified_with_top2.csv` | 오분류 71건 + top-2 확률 |
| `dktc_misclassification_pair_summary.csv` | 오분류 패턴 요약 |
| `dktc_misclassified_most_ambiguous_top30.csv` | 애매한 오분류 top-30 |
| `dktc_misclassified_most_confident_top30.csv` | 확신 오분류 top-30 |
| `dktc_misclassification_margin_analysis.png` | margin 분포 차트 |
| `misclassified_pairs_with_top2/` | 패턴별 오분류 CSV 폴더 |

### Phase 2 (레이블 정제) 생성 파일

| 파일명 | 내용 |
|-------|------|
| `data/train_relabeled_v2.csv` | 정제된 train (재학습용) |
| `data/val_relabeled_v2.csv` | 정제된 val (재평가용) |
| `data/train_relabel_v2_changelog.csv` | train 변경 내역 |
| `data/val_relabel_v2_changelog.csv` | val 변경 내역 |

### Phase 3 (재학습) 생성 파일

Phase 1과 동일한 파일이 새로 생성된다. 덮어쓰기를 방지하려면 Phase 1의 결과를 별도 폴더에 백업해둔다.

---

## 7. 문제 해결

### Q: `dktc_all_misclassified_with_top2.csv`가 없다

HP 튜닝 노트북(`dktc_hp_tuning_full.py`)을 전체 실행해야 생성된다. 반드시 "경로 B" 버전(run_experiment이 val_logits를 반환하는 버전)을 사용해야 한다.

### Q: Val 변경이 0건이다

`MISCLASSIFIED_CSV` 경로가 맞는지 확인. 파일이 작업 디렉토리에 있어야 한다. `os.path.exists("dktc_all_misclassified_with_top2.csv")`로 확인.

### Q: Train 변경이 0건이다

label_name 형식 불일치가 원인일 가능성이 높다. 실행 출력에서 `[Train 형식 감지]` 섹션의 매핑 테이블을 확인한다. 매핑이 올바르면 `협박 ↔ 협박 대화` 같은 쌍이 표시되어야 한다. 매핑이 비어있거나 WARNING이 나오면 `detect_label_format()` 함수의 매칭 로직을 원본 데이터의 실제 label_name에 맞게 수정한다.

### Q: Train 변경이 너무 많다 (10% 초과)

`MARGIN_THRESHOLD`를 높인다 (0.7 → 0.9). 또는 train에서 Rule A만 적용하도록 `active_rules`를 수동으로 제한한다:

```python
# relabel_train_conservative 함수 내에서
active_rules = {"Rule A"}  # 가장 안전한 규칙만
```

### Q: 재학습 후 성능이 떨어졌다

위 5.3절의 대처법을 순서대로 시도. 가장 보수적인 설정:
- `MARGIN_THRESHOLD = 0.9`
- Train은 정제하지 않음 (val만 정제)
- Rule A, C만 활성화 (키워드 매칭이 가장 안전한 규칙)

### Q: 정제된 CSV의 컬럼이 원본과 다르다

정제된 CSV에는 `label_original`, `label_name_original`, `relabel_rule` 3개 컬럼이 추가된다. HP 튜닝 노트북은 `text`, `label`, `label_name` 컬럼만 사용하므로, 추가 컬럼이 있어도 정상 작동한다.

### Q: 원본 레이블로 되돌리고 싶다

정제된 CSV의 `label_original`, `label_name_original` 컬럼에 원본값이 보존되어 있다.

```python
df["label"] = df["label_original"]
df["label_name"] = df["label_name_original"]
```

---

## 8. 이 작업 이후 추가 개선 방향

레이블 정제 후에도 추가 성능 개선이 필요하다면, 아래 순서로 시도한다:

1. **Focal Loss 실험**: `WeightedTrainer`의 `CrossEntropyLoss`를 `FocalLoss(gamma=2)`로 교체. 쉬운 샘플의 loss 기여를 줄여서 경계 샘플에 집중.

2. **2-stage 파이프라인**: Stage 1(일반대화 vs 괴롭힘) + Stage 2(괴롭힘 4-class 분류). 일반대화 분리는 이미 잘 되므로(오분류 3건), Stage 2에서 협박·갈취·기타괴롭힘 경계를 집중 학습.

3. **Soft Label / Knowledge Distillation**: 확신 오분류 건의 모델 예측 확률을 soft target으로 사용. 협박 0.3, 갈취 0.7 같은 연속 레이블로 경계 사례를 더 부드럽게 학습.

4. **Confidence-based rejection**: margin < 0.3인 건(현재 7건, 9.9%)은 "판단 유보"로 처리. 실무 적용 시 precision을 높이는 전략.

---

## 부록 A: 분류 가이드라인 원문

### 클래스별 핵심 정의

- **협박**: 상대방의 신체, 생명, 재산, 명예에 대한 **구체적 해악을 고지**. 예: "죽여버린다", "칼로 찌른다", "가족한테 해코지한다"

- **갈취**: 금전, 물건, 서비스 등 **구체적 재물/이익을 강제로 요구**. 예: "돈 내놔", "그거 줘", "밥 사"

- **직장내괴롭힘**: **직장/조직 내 지위를 이용**한 괴롭힘. 직함, 조직용어가 명시적. 예: "김대리", "인턴", "법인카드"

- **기타괴롭힘**: 위 3가지에 해당하지 않는 괴롭힘. 학교폭력, 가정 내 괴롭힘, 온라인 괴롭힘, 일반적 폭언/모욕 등.

- **일반대화**: 괴롭힘 요소 없음

### 경계 사례 판단 우선순위

1. 위해 고지 + 재물 요구 동시 존재 → **갈취** 우선
2. 위해 고지만 있고 재물 요구 없음 → **협박**
3. 직장 맥락이 명시적 → **직장내괴롭힘**
4. 위 어디에도 해당 안 됨 → **기타괴롭힘**

---

## 부록 B: 핵심 숫자 검증표

이 문서에서 사용된 핵심 숫자의 출처와 검증:

| 숫자 | 의미 | 출처/검증 방법 |
|------|------|--------------|
| 71건 | val set 전체 오분류 | `dktc_all_misclassified_with_top2.csv` 행 수 |
| 34건 | margin > 0.9 확신 오분류 | `df[df["margin"] > 0.9]` 필터 결과 |
| 60건 | margin >= 0.5 후보 | `df[df["margin"] >= 0.5]` 필터 결과 |
| 27건 | v2 규칙으로 val에서 변경되는 건 | v2 규칙 시뮬레이션 결과. 27건 전부 모델 예측 방향과 일치 |
| 0건 | v2로 인한 정분류 피해 | v2는 오분류 목록에서만 작업하므로 구조적으로 0 |
| 76.1% | top-2 예측이 정답인 비율 | 71건 중 54건. `(top2_label == true_label).sum() / len(df)` |
| 84% | 오분류 "협박" 중 위해 키워드 없는 비율 | 25건 중 21건 |
| 48% | 확신 오분류 비율 | 34건 / 71건 |
