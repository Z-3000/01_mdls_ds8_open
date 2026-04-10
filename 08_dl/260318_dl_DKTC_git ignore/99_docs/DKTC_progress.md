# DKTC 프로젝트 진행 현황

> 마지막 업데이트: 2026-03-12  
> 마감: 2026-03-18  
> 환경: Docker (`py-gpu-env`) + VS Code Dev Container

---

## 1. 프로젝트 개요

- **목표**: 한국어 위협 대화 5클래스 분류 모델 구축
- **데이터**: DKTC (Dataset of Korean Threatening Conversations) by TUNiB
- **모델**: `klue/roberta-base`
- **저장 경로**: `./models/dktc_5class/`

### 클래스 구성
| label | 클래스 | 학습 샘플 수 |
|-------|--------|------------|
| 0 | 협박 | 717 |
| 1 | 갈취 | 785 |
| 2 | 직장내괴롭힘 | 783 |
| 3 | 기타괴롭힘 | 875 |
| 4 | 일반대화 | 15 (오버샘플링→200) |

---

## 2. 완료된 작업

### ✅ 01_data_load.ipynb
- DKTC GitHub에서 train.csv 로드
- 라벨 매핑 (4클래스)
- 저장: `data/train.csv`

### ✅ 02_model_train.ipynb
- 4클래스 모델 학습 (F1 Macro 0.895)
- threshold 실험 → 일반대화 처리 한계 확인
- 일반대화 15개 직접 작성 → 오버샘플링 200개
- **WeightedTrainer** + class weight 적용
- 5클래스 재학습 → **F1 Macro 0.911**
- 일반대화 precision/recall **1.00**
- 모델 저장: `./models/dktc_5class/`

### ✅ 03_inference.ipynb
- `predict(text)` 함수 완성
- 정상 추론 확인

### ✅ 오류 분석
```
갈취 → 기타괴롭힘 (13)  ← 가장 많음
협박 → 갈취 (13)        ← 가장 많음
협박 → 기타괴롭힘 (11)
갈취 → 협박 (7)
```
**인사이트**: 갈취↔협박 상호 혼동 (언어적 경계 모호), 직장내괴롭힘은 도메인 어휘 덕분에 오류 적음

### ✅ AI Hub 데이터 압축 해제
- `020.주제별 텍스트 일상 대화 데이터` → zip 26개 해제 완료
- `044.페르소나 대화` → 미완료
- `011.일상대화 한국어 멀티세션 데이터` → 파일 없음 (재다운로드 필요)

### ✅ 020 데이터 구조 파악
- JSON 파일 98,652개
- 구조: `data["info"][0]["annotations"]["text"]`
- text 형식: `"1 : 발화내용\n2 : 발화내용\n..."` (문자열)
- 대화 추출 코드 작성 완료 (미실행)

---

## 3. 지금 해야 할 작업 (순서대로)

### STEP 1: 020 대화 추출 실행

```python
import json, os, pandas as pd, re

base = "/home/jovyan/d/git/01_mdls_ds8_open/08_dl/260318_dl_DKTC/00_data"
target = os.path.join(base, "020.주제별 텍스트 일상 대화 데이터", "01.데이터", "1.Training")

json_files = []
for root, dirs, files in os.walk(target):
    if "라벨링데이터" in root:
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

conversations = []
for fpath in json_files:
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for info in data["info"]:
            raw_text = info["annotations"]["text"]
            lines = raw_text.strip().split("\n")
            cleaned = [re.sub(r"^\d+\s*:\s*", "", line.strip()) for line in lines]
            cleaned = [c for c in cleaned if c]
            full_conv = " ".join(cleaned)
            if len(full_conv) > 10:
                conversations.append(full_conv)
    except:
        continue

print(f"추출된 대화 수: {len(conversations)}")
```

### STEP 2: DataFrame 변환 + 샘플링

```python
normal_020 = pd.DataFrame({
    "conversation": conversations,
    "label": 4,
    "label_name": "일반 대화"
})

# 500개 샘플링 (기존 위협 클래스 700~900개 수준에 맞춤)
normal_sampled = normal_020.sample(n=500, random_state=42)
print(normal_sampled.shape)
```

### STEP 3: 기존 DKTC와 합치기

```python
# train_df는 기존 4클래스 데이터 (3950개)
combined = pd.concat([train_df, normal_sampled], ignore_index=True)
print(combined["label_name"].value_counts())
# 이번엔 오버샘플링 불필요 - 500개면 충분
```

### STEP 4: 5클래스 재학습

- 이전 `WeightedTrainer` 코드 그대로 사용
- 오버샘플링 제거 (일반대화 500개이므로 불필요)
- class weight 재계산 필요 (비율 바뀌었으므로)
- 기대 성능: F1 0.91 이상 (일반대화 다양성 증가로 일반화 향상)

### STEP 5: OOD(분포 외) 테스트

```python
ood_tests = [
    "야 너 나중에 두고 봐",
    "ㅋㅋ 그러다 큰일남",
    "진짜 죽겠다 이 더위에",
    "돈 좀 빌려줄 수 있어?",
    "너 그날 일 잊지 마",
    "가만 안 놔둘 줄 알아 ㅋ",
]
for text in ood_tests:
    result = predict(text)
    print(f"{text} → {result['label']} ({result['confidence']*100:.1f}%)")
```

---

## 4. 선택적 추가 작업 (시간 있을 때)

| 작업 | 예상 시간 | 효과 |
|------|----------|------|
| Confusion Matrix 시각화 | 30분 | 포트폴리오용 |
| 토큰 중요도 시각화 (transformers-interpret) | 1시간 | 모델 해석력 |
| 044 페르소나 데이터 압축 해제 + 추가 학습 | 2시간 | 성능 향상 |
| Gradio 데모 | 30분 | 시연용 |
| README 작성 | 1시간 | 필수 |

---

## 5. 환경 정보

```
Docker 컨테이너: py-gpu-env
GPU: RTX 5060 Ti 16GB
D드라이브 마운트: /home/jovyan/d
HuggingFace 캐시: /root/.cache/huggingface
모델 저장: ./models/dktc_5class/
```

### 데이터 경로 (Docker 기준)
```
/home/jovyan/d/git/01_mdls_ds8_open/08_dl/260318_dl_DKTC/
├── 00_data/
│   ├── 020.주제별 텍스트 일상 대화 데이터/  ← zip 해제 완료
│   ├── 044.페르소나 대화/                  ← zip 해제 필요
│   └── 011.일상대화 한국어 멀티세션 데이터/  ← 재다운로드 필요
├── data/
│   └── train.csv
└── models/
    └── dktc_5class/                       ← 현재 최종 모델
```

---

## 6. 현재 모델 성능 요약

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| 협박 | 0.91 | 0.83 | 0.86 |
| 갈취 | 0.86 | 0.83 | 0.84 |
| 직장내괴롭힘 | 0.93 | 0.97 | 0.95 |
| 기타괴롭힘 | 0.86 | 0.91 | 0.89 |
| 일반대화 | 1.00 | 1.00 | 1.00 |
| **macro avg** | **0.91** | **0.91** | **0.91** |

> ⚠️ 일반대화 1.00은 오버샘플링(15개 기반)의 한계. 020 데이터로 재학습 후 검증 필요.


검증 결과 (5회 교차 확인)
✅ 정확한 내용들
항목컴팩트 요약트랜스크립트 확인원본 4클래스 총 3950건✅GitHub README: 896+981+979+1094=3950학습/검증 split 3160/790✅실제 로그 확인F1 Macro 4클래스 0.895✅실제 로그 확인F1 Macro 5클래스 0.911✅실제 로그 확인5클래스 support 합계 672✅143+157+157+175+40=672일반대화 15개→오버샘플링 200개✅코드 확인WeightedTrainer + class_weight✅코드 확인batch=32, epochs=5, fp16=True✅코드 확인직장내괴롭힘 오류 최소 4건✅트랜스크립트 확인

❌ 발견된 오류 2건
오류 1 — 오류 분석 항목 날조
컴팩트 요약에 기재된 항목:
기타괴롭힘→갈취 (9)     ← 트랜스크립트 어디에도 없음
갈취→직장내괴롭힘 (7)   ← 트랜스크립트 어디에도 없음
트랜스크립트 실제 기록:
갈취 → 기타괴롭힘 (13)  ★
협박 → 갈취 (13)        ★
협박 → 기타괴롭힘 (11)
갈취 → 협박 (7)
즉 총 4쌍인데, 컴팩트 요약이 이 중 갈취→협박(7)을 맞게 적었지만 존재하지 않는 2쌍을 만들어냈음.
오류 2 — "최적 epoch 3" 근거 없음
컴팩트 요약: "최적 epoch 3: F1 Macro 0.911"
트랜스크립트 3개 전체에서 best epoch이 몇 번인지 명시된 곳 없음. load_best_model_at_end=True로 자동 선택된 것은 맞지만 epoch 번호는 확인 불가.

⚠️ 미검증 (트랜스크립트에 출력값 없음)

class weights [0.937, 0.856, 0.859, 0.768, 3.36] — 계산 코드는 있으나 실제 출력 print 결과가 트랜스크립트에 없어 확인 불가


결론: 컴팩트 요약은 대부분 정확하나, 오류 분석 6개 항목 중 2개가 사실과 다른 내용임. 실제 오류 패턴은 4쌍만 기록되어 있었음.