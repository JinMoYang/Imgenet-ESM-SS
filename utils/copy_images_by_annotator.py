# copy_images_by_annotator.py

import pandas as pd
import shutil
import os

def copy_images_by_annotator(csv_path, source_dir, output_prefix):
    """
    CSV 파일을 읽어 annotator별로 이미지를 복사한다.
    
    Args:
        csv_path: CSV 파일 경로
        source_dir: 원본 이미지 폴더 경로
        output_prefix: 출력 폴더 접두사
    """
    df = pd.read_csv(csv_path)
    
    # annotator별 이미지 수집
    annotator_images = {}
    for _, row in df.iterrows():
        image_id = row['ImageID']
        for col in ['annotator1', 'annotator2', 'annotator3']:
            name = row[col]
            if pd.notna(name):  # 빈 값 제외
                if name not in annotator_images:
                    annotator_images[name] = []
                annotator_images[name].append(image_id)
    
    # annotator별로 폴더 생성 및 이미지 복사
    for name, images in annotator_images.items():
        output_dir = f"{output_prefix}_{name}"
        os.makedirs(output_dir, exist_ok=True)
        
        for image_id in images:
            src = os.path.join(source_dir, image_id)
            dst = os.path.join(output_dir, image_id)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"경고: 파일을 찾을 수 없습니다 - {src}")
        
        print(f"{name}: {len(images)}개 이미지 복사 완료 -> {output_dir}")

if __name__ == '__main__':
    copy_images_by_annotator('batch_1st.csv', 'sampled_images', 'batch_1st')
