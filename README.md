# ImageNet ESM Segmentation - Annotation 수집 프로젝트

## 📋 목차
- [문제 배포](#문제-배포)
- [Protocol](#protocol)
  - [Phase 1](#phase-1-배포)
  - [Phase 2](#phase-2-배포-준비)
  - [시각화 프로세스](#시각화-프로세스)
- [Feedback 정합과정](#feedback-정합과정)
- [최종 피드백 반영](#최종-피드백-반영)
- [File Structure](#file-structure)
- [추가 유틸](#추가-유틸)
- [구글 드라이브 필수 파일](#구글-드라이브에-있어야-하는것)

---

## 문제 배포

### CSV 파일 구조
```csv
ImageID,annotator1,annotator2,annotator3,Reviewer 1,Reviewer 2,Reviewer 3
ILSVRC2012_val_00022657.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023031.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023088.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023314.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023359.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023463.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00023617.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00023828.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00024165.JPEG,우진,은수,리안,홍준,다연,
```

### 사진 분류
- **스크립트**: `copy_images_by_annotator.py`
- **필수 CSV 구조**: `ImageID, annotator1, annotator2, annotator3, Reviewer1, Reviewer2, Reviewer3`

---

## Protocol

### Phase 1 배포

#### 1️⃣ Annotation 수집 및 정리
1. **결과 파일 정리**
   - `batch_1st_phase1_result` 파일에 `batch_{num_string}_{name}_annotation` 정리

2. **전처리**
   - ~~`rename_remove_prefix.py`: 접두사 숫자 제거 (필요없음)~~
   - **`add_annotator_to_json.py`**: Infinity polygon 제거 및 annotator 추가
     > ⚠️ 서버 실행 시 infinity polygon이 섞일 수 있으므로 한 번 실행 권장

3. **JSON 분배**
   - **`distribute_json.py`**: annotator1, annotator2, annotator3 폴더로 분배

4. **Annotation 정합**
   - **`code_final_final/merge_annotations.py`**: 어노테이션 정합

5. **리뷰 및 피드백**
   - Google Form Review 진행
   - 구글 스프레드시트 앱스크립트로 피드백 작성 (시각화 부분 참고)

---

### Phase 2 배포 준비

#### 1️⃣ Phase 2 준비
1. **폴더 및 이미지 분배**
   - **스크립트**: `distribute_annotations.py`
   - Phase 1의 `merge_annotations`에 있는 JSON 파일들을 `batch_1st_reviewers.csv`의 annotator에 맞게 폴더 생성 (보통 5개)
   - 이미지도 JSON에 맞춰 자동 추가
   - **CSV 처리**: 스프레드시트에서 App Script로 positive row만 남기기

#### 2️⃣ Phase 2 진입
1. **결과 파일 정리**
   - `batch_1st_phase2_result` 파일에 `batch_{num_string}_{name}_annotation` 정리

2. **전처리**
   - **`rename_remove_prefix.py`**: 접두사 숫자 제거
   - **`add_annotator_to_json.py`**: Infinity polygon 제거 및 annotator 추가

3. **JSON 분배**
   - **`distribute_json.py`**: `batch_1st.csv` 기반으로 annotator1, 2, 3으로 매핑
   - 각 폴더명: `batch_{num_string}_{name}_annotation`
   - 중복 없이 annotator1부터 순차적으로 저장

4. **Annotation 정합**
   - **`code_final_final`**: 어노테이션 정합 과정

5. **리뷰 및 피드백**
   - Google Form Review 진행
   - 구글 스프레드시트 앱스크립트로 피드백 작성

---

### 시각화 프로세스

1. **이미지 매칭**
   - **스크립트**: `match_json_images.py`
   - `merged_annotations` 폴더의 JSON과 동일한 이름의 이미지를 `sampled_images`에서 찾아 `merged_annotations_with_images` 폴더 생성

2. **시각화 실행**
   - **스크립트**: `visualization_all_test.py`

3. **구글 드라이브 업로드**
   - `visualized` 폴더의 이미지들 + `merged_annotations`의 JSON들을 `merged_images_phase#`에 업로드

#### 구글 드라이브 구조
```
메인 디렉토리/
├── original_images/          # 원본 이미지
├── merged_images_phase1/     # 시각화 이미지 + JSON
└── merged_images_phase2/
```

- **필요한 설정**
  - `phase1_외부배치_구글폼`, `phase2_구글폼` 실행
  - Apps Script에서 상위 폴더 지정: `original_images`, `merged_image_phase1`, `merged_image_phase2`
  - 리뷰어 CSV 파일 업로드 (포맷 확인 필요)

---


## Feedback 정합과정

### Phase별 피드백 처리
- **Phase 1**: Positive + Negative 모두 포함
- **Phase 2**: Positive만 포함
  - 스프레드시트에서 App Script로 positive row만 필터링

### 프로세스

#### 1️⃣ 스프레드시트 생성 및 이미지-드라이브 매핑
- 원본 스프레드시트를 복사하여 새 스프레드시트 생성
- 이미지 링크를 자동으로 추가

#### 2️⃣ 리뷰어별 리뷰 매핑
- 생성된 스프레드시트와 리뷰어별 폼 데이터를 결합
- 각 리뷰어당 5개 열 생성 (첫 번째 제외, 4개 열 복사)

#### 3️⃣ Reviewer 정보 압축
- Reviewer 1, 2, 3의 정보를 한 줄로 압축하여 정리

---

## 최종 피드백 반영

### 이미지 매칭 및 생성
**스크립트**: `match_json_images.py`

- `merged_annotations` 폴더의 JSON 파일과 동일한 이름의 이미지를 `sampled_images`에서 찾아 `merged_annotations_with_images` 폴더 생성

#### 세부 로직
1. **1번 폴더**: `.json` 파일들 위치
2. **2번 폴더**: 이미지 파일들 위치
3. **매칭 프로세스**:
   - JSON 파일 이름과 동일한 이름의 이미지를 2번 폴더에서 검색
   - 매칭되는 이미지들을 3번 폴더(지정 가능)에 복사 또는 이동
   - 폴더명과 위치는 사용자 지정 가능

---

## File Structure

### 필수 CSV 파일 구조

#### `batch_1st_reviewers.csv`
```csv
ImageID, annotator1, annotator2, annotator3, Reviewer1, Reviewer2, Reviewer3
```

#### `feedback_merge.csv`
```csv
image_name, annotators(띄어쓰기로 구분)
```

---

## 추가 유틸

### 유틸리티 스크립트 목록

| 스크립트 | 기능 |
|---------|------|
| `json_to_jpeg_csv.py` | 폴더의 `.json` 파일을 찾아 `.JPEG`으로 변환하고 ImageID만 포함된 CSV 생성 |
| `remove_jpeg_files.py` | 폴더 내 모든 `.JPEG` 파일 제거 |

---

## 구글 드라이브에 있어야 하는것

### 필수 파일 및 폴더

1. **이미지 소스** (원본 이미지 폴더)
2. **batch_1st_reviewers.csv**
   ```csv
   ImageID, annotator1, annotator2, annotator3, Reviewer1, Reviewer2, Reviewer3
   ```
