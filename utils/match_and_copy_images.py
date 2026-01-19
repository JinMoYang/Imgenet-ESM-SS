# scripts/match_and_copy_images.py

import os
import shutil
from pathlib import Path
from typing import Set, List, Tuple

"""
이미지 파일이 있는 1번 폴더에서, 2번 폴더의 JSON 파일명과 일치하는 이미지들을 찾아 2번 폴더로 복사하는 스크립트
"""

# 지원하는 이미지 확장자


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}


def get_json_basenames(folder_path: Path) -> Set[str]:
    """
    폴더 내 모든 JSON 파일의 basename을 추출한다.
    
    Args:
        folder_path: JSON 파일들이 있는 폴더 경로
        
    Returns:
        확장자를 제외한 JSON 파일명 집합
    """
    json_files = folder_path.glob('*.json')
    return {f.stem.lower() for f in json_files}


def find_matching_images(source_folder: Path, target_basenames: Set[str]) -> List[Path]:
    """
    소스 폴더에서 target_basenames와 일치하는 이미지 파일을 찾는다.
    
    Args:
        source_folder: 이미지 파일을 검색할 폴더
        target_basenames: 매칭할 파일명 집합 (확장자 제외)
        
    Returns:
        매칭된 이미지 파일의 경로 리스트
    """
    matched_images = []
    
    for file_path in source_folder.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            basename = file_path.stem.lower()
            
            if ext in IMAGE_EXTENSIONS and basename in target_basenames:
                matched_images.append(file_path)
    
    return matched_images


def copy_images(image_paths: List[Path], dest_folder: Path) -> Tuple[int, List[str]]:
    """
    이미지 파일들을 목적지 폴더로 복사한다.
    
    Args:
        image_paths: 복사할 이미지 경로 리스트
        dest_folder: 복사 목적지 폴더
        
    Returns:
        (성공한 파일 수, 실패한 파일명 리스트)
    """
    success_count = 0
    failed_files = []
    
    for img_path in image_paths:
        try:
            dest_path = dest_folder / img_path.name
            
            # 이미 존재하는 경우 덮어쓰기
            shutil.copy2(img_path, dest_path)
            success_count += 1
            print(f"복사 완료: {img_path.name}")
            
        except Exception as e:
            failed_files.append(f"{img_path.name} (오류: {str(e)})")
            print(f"복사 실패: {img_path.name} - {e}")
    
    return success_count, failed_files


def match_and_copy_images(source_folder: str, target_folder: str) -> None:
    """
    소스 폴더에서 타겟 폴더의 JSON 파일명과 일치하는 이미지를 찾아 복사한다.
    
    Args:
        source_folder: 이미지 파일들이 있는 폴더 (1번 폴더)
        target_folder: JSON 파일들이 있고 이미지를 복사할 폴더 (2번 폴더)
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    
    # 폴더 존재 여부 검증
    if not source_path.exists():
        raise FileNotFoundError(f"소스 폴더가 존재하지 않습니다: {source_folder}")
    if not target_path.exists():
        raise FileNotFoundError(f"타겟 폴더가 존재하지 않습니다: {target_folder}")
    
    print(f"작업 시작")
    print(f"소스 폴더: {source_path.absolute()}")
    print(f"타겟 폴더: {target_path.absolute()}")
    print("-" * 60)
    
    # 1단계: JSON 파일명 추출
    json_basenames = get_json_basenames(target_path)
    print(f"타겟 폴더에서 {len(json_basenames)}개의 JSON 파일을 발견했습니다.")
    
    if not json_basenames:
        print("JSON 파일이 없어 작업을 종료합니다.")
        return
    
    # 2단계: 매칭되는 이미지 검색
    matched_images = find_matching_images(source_path, json_basenames)
    print(f"소스 폴더에서 {len(matched_images)}개의 매칭 이미지를 발견했습니다.")
    print("-" * 60)
    
    if not matched_images:
        print("매칭되는 이미지가 없습니다.")
        return
    
    # 3단계: 이미지 복사
    success_count, failed_files = copy_images(matched_images, target_path)
    
    # 결과 보고
    print("-" * 60)
    print(f"작업 완료: {success_count}/{len(matched_images)}개 파일 복사 성공")
    
    if failed_files:
        print(f"\n실패한 파일 ({len(failed_files)}개):")
        for failed in failed_files:
            print(f"  - {failed}")


if __name__ == "__main__":
    # 사용 예시
    SOURCE_FOLDER = "path/to/folder1"  # 이미지가 있는 폴더
    TARGET_FOLDER = "path/to/folder2"  # JSON이 있고 이미지를 추가할 폴더
    
    try:
        match_and_copy_images(SOURCE_FOLDER, TARGET_FOLDER)
    except Exception as e:
        print(f"오류 발생: {e}")