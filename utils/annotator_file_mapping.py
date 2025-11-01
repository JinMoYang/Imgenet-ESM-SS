import pandas as pd
import shutil
from pathlib import Path
import sys

# --- 1. 설정 (Configuration) ---

# CSV 파일 경로
CSV_FILE = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st.csv'

# 원본 어노테이션 폴더가 들어있는 기본 폴더
# e.g., 'batch_1st_result/batch_1st_우진_annotation/...'
SOURCE_BASE_DIR = Path('/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_result')

# 파일을 옮길 대상 기본 폴더 (현재 폴더를 기준으로)
DEST_BASE_DIR = Path('/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation') 

# CSV에서 읽어올 어노테이터 컬럼 이름과
# 매칭되는 대상 폴더 이름 (동일하게 설정)
ANNOTATOR_COLS = ['annotator1', 'annotator2', 'annotator3']
DEST_FOLDERS = ['annotator1', 'annotator2', 'annotator3']

# --- 2. 대상 폴더 생성 ---
print(f"'{DEST_BASE_DIR.resolve()}' 위치에 대상 폴더를 생성합니다...")
for folder in DEST_FOLDERS:
    (DEST_BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# --- 3. CSV 파일 읽기 ---
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"오류: CSV 파일을 찾을 수 없습니다. (경로: {CSV_FILE})")
    sys.exit(1)
except Exception as e:
    print(f"오류: CSV 파일을 읽는 중 문제가 발생했습니다: {e}")
    sys.exit(1)

print(f"'{CSV_FILE}' 파일을 성공적으로 읽었습니다. 총 {len(df)}개 항목 처리 시작...")

# --- 4. 파일 복사/이동 로직 ---
copied_count = 0
missing_count = 0

# DataFrame의 각 행(row)을 순회합니다.
for index, row in df.iterrows():
    image_id_ext = row['ImageID']
    
    # ImageID가 비어있는 경우 (NaN) 스킵
    if pd.isna(image_id_ext):
        print(f"경고: {index+2}번째 행은 ImageID가 비어있어 건너뜁니다.")
        continue
        
    # ImageID에서 확장자(.JPEG)를 제거하고 .json을 붙여 파일명을 만듭니다.
    # e.g., 'ILSVRC2012_val_00000371.JPEG' -> 'ILSVRC2012_val_00000371'
    image_stem = Path(image_id_ext).stem 
    json_filename = f"{image_stem}.json"

    # 'annotator1', 'annotator2', 'annotator3' 컬럼을 순회
    for col_name in ANNOTATOR_COLS:
        # 해당 컬럼의 어노테이터 이름 (e.g., '우진', '은수', NaN)
        annotator_name = row[col_name]
        
        # 어노테이터 이름이 비어있는 경우 (NaN) 스킵
        if pd.isna(annotator_name):
            continue
        
        # 1. 원본 파일 경로 생성
        # e.g., batch_1st_result/batch_1st_우진_annotation/ILSVRC2012_val_00000371.json
        source_folder_name = f"batch_1st_{annotator_name}_annotation"
        source_path = SOURCE_BASE_DIR / source_folder_name / json_filename
        
        # 2. 대상 파일 경로 생성
        # e.g., ./annotator1/ILSVRC2012_val_00000371.json
        dest_path = DEST_BASE_DIR / col_name / json_filename
        
        # 3. 파일 복사 (shutil.copy)
        #    파일을 완전히 '이동'하고 원본을 지우려면
        #    shutil.copy(source_path, dest_path) 대신
        #    shutil.move(source_path, dest_path) 를 사용하세요.
        
        if source_path.exists():
            try:
                shutil.copy(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"오류: {source_path} 복사 중 문제 발생: {e}")
        else:
            # CSV에는 이름이 있지만 실제 파일이 없는 경우
            print(f"경고: 원본 파일을 찾을 수 없습니다: {source_path}")
            missing_count += 1

print("\n--- 작업 완료 ---")
print(f"총 {copied_count}개의 파일을 성공적으로 복사했습니다.")
if missing_count > 0:
    print(f"총 {missing_count}개의 원본 파일을 찾지 못했습니다 (위 경고 메시지 확인).")