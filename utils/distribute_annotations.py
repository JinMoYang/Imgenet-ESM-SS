# distribute_annotations.py

# distribute_annotations.py

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Set


def extract_unique_annotators(csv_path: str) -> Set[str]:
    """
    CSV에서 annotator1, annotator2, annotator3 열의 고유 어노테이터 목록을 추출합니다.
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        고유 어노테이터 이름 집합
    """
    df = pd.read_csv(csv_path)
    annotator_columns = ['annotator1', 'annotator2', 'annotator3']
    
    annotators = set()
    for col in annotator_columns:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            annotators.update(unique_values)
    
    return annotators


def get_annotator_images(csv_path: str, annotator_name: str) -> List[str]:
    """
    특정 어노테이터가 담당한 ImageID 목록을 반환합니다.
    
    Args:
        csv_path: CSV 파일 경로
        annotator_name: 어노테이터 이름
        
    Returns:
        ImageID 목록
    """
    df = pd.read_csv(csv_path)
    annotator_columns = ['annotator1', 'annotator2', 'annotator3']
    
    mask = df[annotator_columns].apply(
        lambda row: annotator_name in row.values, axis=1
    )
    image_ids = df.loc[mask, 'ImageID'].tolist()
    
    return image_ids


def copy_json_files(
    json_dir: str,
    csv_path: str,
    output_prefix: str
) -> dict:
    """
    1단계: 어노테이터별로 JSON 파일을 복사합니다.
    
    Args:
        json_dir: JSON 파일이 있는 폴더 경로
        csv_path: CSV 파일 경로
        output_prefix: 출력 폴더 이름의 접두사
        
    Returns:
        어노테이터별 출력 폴더 경로 딕셔너리
    """
    json_path = Path(json_dir)
    annotators = extract_unique_annotators(csv_path)
    
    print(f"고유 어노테이터 수: {len(annotators)}")
    print(f"어노테이터 목록: {sorted(annotators)}")
    
    annotator_dirs = {}
    
    for annotator in annotators:
        output_dir = Path(f"{output_prefix}_{annotator}")
        output_dir.mkdir(exist_ok=True)
        annotator_dirs[annotator] = output_dir
        
        image_ids = get_annotator_images(csv_path, annotator)
        print(f"\n어노테이터 '{annotator}': {len(image_ids)}개 이미지 할당됨")
        
        copied_jsons = 0
        missing_jsons = []
        
        for image_id in image_ids:
            json_filename = image_id.replace('.JPEG', '.json')
            src_json = json_path / json_filename
            
            if src_json.exists():
                dst_json = output_dir / json_filename
                shutil.copy2(src_json, dst_json)
                copied_jsons += 1
            else:
                missing_jsons.append(json_filename)
        
        print(f"  복사된 JSON: {copied_jsons}/{len(image_ids)}")
        
        if missing_jsons:
            print(f"  누락된 JSON: {len(missing_jsons)}개")
            if len(missing_jsons) <= 5:
                print(f"    {missing_jsons}")
    
    return annotator_dirs


def copy_images_from_json(
    image_dir: str,
    annotator_dirs: dict
) -> None:
    """
    2단계: 각 폴더의 JSON 파일명 기반으로 이미지를 복사합니다.
    
    Args:
        image_dir: 이미지 파일이 있는 폴더 경로
        annotator_dirs: 어노테이터별 출력 폴더 경로 딕셔너리
    """
    image_path = Path(image_dir)
    
    print("\n=== 이미지 복사 시작 ===")
    
    for annotator, output_dir in annotator_dirs.items():
        json_files = list(output_dir.glob("*.json"))
        print(f"\n어노테이터 '{annotator}': {len(json_files)}개 JSON 파일 발견")
        
        copied_images = 0
        missing_images = []
        
        for json_file in json_files:
            image_filename = json_file.stem + '.JPEG'
            src_image = image_path / image_filename
            
            if src_image.exists():
                dst_image = output_dir / image_filename
                shutil.copy2(src_image, dst_image)
                copied_images += 1
            else:
                missing_images.append(image_filename)
        
        print(f"  복사된 이미지: {copied_images}/{len(json_files)}")
        
        if missing_images:
            print(f"  누락된 이미지: {len(missing_images)}개")
            if len(missing_images) <= 5:
                print(f"    {missing_images}")


def distribute_files(
    json_dir: str,
    image_dir: str,
    csv_path: str,
    output_prefix: str
) -> None:
    """
    어노테이터별로 JSON과 이미지 파일을 분류합니다.
    
    Args:
        json_dir: JSON 파일이 있는 폴더 경로
        image_dir: 이미지 파일이 있는 폴더 경로
        csv_path: CSV 파일 경로
        output_prefix: 출력 폴더 이름의 접두사
    """
    # 1단계: JSON 파일 복사
    print("=== 1단계: JSON 파일 복사 ===")
    annotator_dirs = copy_json_files(json_dir, csv_path, output_prefix)
    
    # 2단계: 이미지 파일 복사
    print("\n=== 2단계: 이미지 파일 복사 ===")
    copy_images_from_json(image_dir, annotator_dirs)
    
    print("\n=== 완료 ===")


if __name__ == "__main__":
    # 사용 예시
    json_directory = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_validation/merged_annotations"
    image_directory = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/sampled_images"
    csv_file = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_question/batch_3rd_phase2.csv"
    output_prefix = "batch_test_phase2"
    
    distribute_files(
        json_dir=json_directory,
        image_dir=image_directory,
        csv_path=csv_file,
        output_prefix=output_prefix
    )
