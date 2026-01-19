# file_organizer.py

import pandas as pd
import shutil
from pathlib import Path
import sys

# --- 1. 설정 (Configuration) ---

# CSV 파일 경로
CSV_FILE = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st.csv'

# 원본 어노테이션 폴더들이 들어있는 기본 폴더
SOURCE_BASE_DIR = Path('/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_result')

# 파일을 옮길 대상 기본 폴더
DEST_BASE_DIR = Path('/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation')

# --- 2. CSV 파일 읽기 ---
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"오류: CSV 파일을 찾을 수 없습니다. (경로: {CSV_FILE})")
    sys.exit(1)
except Exception as e:
    print(f"오류: CSV 파일을 읽는 중 문제가 발생했습니다: {e}")
    sys.exit(1)

print(f"'{CSV_FILE}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개 항목 처리 시작...")

# --- 3. 이미지 컬럼명 확인 ---
# 'ImageID' 또는 'image_name' 중 존재하는 컬럼 사용
if 'ImageID' in df.columns:
    image_col = 'ImageID'
elif 'image_name' in df.columns:
    image_col = 'image_name'
else:
    print("오류: CSV에 'ImageID' 또는 'image_name' 컬럼이 없습니다.")
    sys.exit(1)

print(f"이미지 정보는 '{image_col}' 컬럼에서 읽습니다.")

# --- 4. 어노테이터 목록 수집 및 대상 폴더 생성 ---
# CSV 전체에서 등장하는 모든 어노테이터 이름을 수집
all_annotators = set()
for index, row in df.iterrows():
    annotators_str = row.get('annotators', '')
    if pd.notna(annotators_str):
        names = annotators_str.split()
        all_annotators.update(names)

if not all_annotators:
    print("경고: CSV에서 어노테이터를 찾을 수 없습니다. 'annotators' 컬럼을 확인하세요.")
    sys.exit(1)

print(f"발견된 어노테이터: {sorted(all_annotators)}")

# 대상 폴더 생성
print(f"'{DEST_BASE_DIR.resolve()}' 위치에 대상 폴더를 생성합니다...")
for annotator_name in all_annotators:
    (DEST_BASE_DIR / annotator_name).mkdir(parents=True, exist_ok=True)

# --- 5. 원본 폴더 매핑 생성 (각 어노테이터별 원본 폴더 경로) ---
def find_annotation_folder(base_dir, annotator_name):
    """
    주어진 base_dir에서 특정 어노테이터의 annotation 폴더를 찾는다.
    패턴:
      - batch_*_{annotator_name}_annotation
      - feedback_{annotator_name}_annotation
    """
    # batch_* 패턴
    batch_pattern = f"batch_*_{annotator_name}_annotation"
    batch_matches = list(base_dir.glob(batch_pattern))
    
    # feedback 패턴
    feedback_pattern = f"feedback_{annotator_name}_annotation"
    feedback_matches = list(base_dir.glob(feedback_pattern))
    
    all_matches = batch_matches + feedback_matches
    
    if len(all_matches) == 0:
        return None
    elif len(all_matches) == 1:
        return all_matches[0]
    else:
        # 여러 개 발견된 경우 경고 후 첫 번째 사용
        print(f"경고: '{annotator_name}'에 대해 {len(all_matches)}개의 폴더가 발견되었습니다. "
              f"첫 번째 폴더를 사용합니다: {all_matches[0]}")
        return all_matches[0]

annotator_folder_map = {}
for annotator_name in all_annotators:
    folder = find_annotation_folder(SOURCE_BASE_DIR, annotator_name)
    if folder:
        annotator_folder_map[annotator_name] = folder
        print(f"  '{annotator_name}' -> {folder.name}")
    else:
        print(f"경고: '{annotator_name}'에 대한 annotation 폴더를 찾을 수 없습니다.")

# --- 6. 파일 복사 로직 ---
copied_count = 0
missing_count = 0

for index, row in df.iterrows():
    image_id_ext = row[image_col]
    
    # 이미지 ID가 비어있는 경우 스킵
    if pd.isna(image_id_ext):
        print(f"경고: {index+2}번째 행은 '{image_col}'이 비어있어 건너뜁니다.")
        continue
    
    # JSON 파일명 생성 (확장자 제거 후 .json 추가)
    image_stem = Path(image_id_ext).stem
    json_filename = f"{image_stem}.json"
    
    # 'annotators' 컬럼에서 공백으로 구분된 이름들 추출
    annotators_str = row.get('annotators', '')
    if pd.isna(annotators_str):
        continue
    
    annotator_names = annotators_str.split()
    
    for annotator_name in annotator_names:
        # 원본 폴더 경로 확인
        if annotator_name not in annotator_folder_map:
            print(f"경고: '{annotator_name}'에 대한 원본 폴더를 찾을 수 없어 건너뜁니다.")
            continue
        
        source_folder = annotator_folder_map[annotator_name]
        source_path = source_folder / json_filename
        
        # 대상 경로
        dest_path = DEST_BASE_DIR / annotator_name / json_filename
        
        # 파일 복사
        if source_path.exists():
            try:
                shutil.copy(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"오류: {source_path} 복사 중 문제 발생: {e}")
        else:
            print(f"경고: 원본 파일을 찾을 수 없습니다: {source_path}")
            missing_count += 1

print("\n--- 작업 완료 ---")
print(f"총 {copied_count}개의 파일을 성공적으로 복사했습니다.")
if missing_count > 0:
    print(f"총 {missing_count}개의 원본 파일을 찾지 못했습니다 (위 경고 메시지 확인).")