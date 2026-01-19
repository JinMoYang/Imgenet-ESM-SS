# match_json_images.py

import argparse
from pathlib import Path
import shutil
from typing import List, Tuple, Optional


def copy_directory_contents(src_dir: Path, dest_dir: Path) -> int:
    """
    소스 디렉토리의 모든 파일을 목적지 디렉토리로 복사한다.
    
    Args:
        src_dir: 소스 디렉토리 경로
        dest_dir: 목적지 디렉토리 경로
        
    Returns:
        복사된 파일 수
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files = [f for f in src_dir.iterdir() if f.is_file()]
    
    for file_path in files:
        dest_path = dest_dir / file_path.name
        shutil.copy2(file_path, dest_path)
    
    return len(files)


def get_json_files(directory: Path) -> List[Path]:
    """
    지정된 디렉토리에서 모든 .json 파일을 반환한다.
    
    Args:
        directory: JSON 파일이 있는 디렉토리 경로
        
    Returns:
        .json 파일 경로의 리스트
    """
    if not directory.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    json_files = list(directory.glob("*.json"))
    return json_files


def find_matching_image(stem: str, image_dir: Path) -> Optional[Path]:
    """
    주어진 stem과 일치하는 이미지 파일을 찾는다.
    
    Args:
        stem: 확장자를 제외한 파일명
        image_dir: 이미지 파일이 있는 디렉토리 경로
        
    Returns:
        찾은 이미지 파일 경로, 없으면 None
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"]
    
    for ext in image_extensions:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    
    return None


def match_and_copy_images(
    json_dir: Path,
    image_dir: Path,
    output_dir: Path
) -> Tuple[int, int, int]:
    """
    1번 폴더의 모든 파일을 출력 디렉토리에 복사하고,
    JSON 파일과 매칭되는 이미지를 2번 폴더에서 찾아 추가로 복사한다.
    
    Args:
        json_dir: JSON 파일이 있는 디렉토리 (1번 폴더)
        image_dir: 이미지 파일이 있는 디렉토리 (2번 폴더)
        output_dir: 매칭된 파일들을 저장할 디렉토리 (3번 폴더)
        
    Returns:
        (1번 폴더에서 복사된 파일 수, 매칭된 이미지 수, 매칭되지 않은 JSON 수)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1단계: 1번 폴더의 모든 파일을 출력 폴더로 복사
    print("1번 폴더의 모든 파일을 출력 폴더로 복사합니다...")
    copied_count = copy_directory_contents(json_dir, output_dir)
    print(f"총 {copied_count}개의 파일을 복사했습니다.\n")
    
    # 2단계: 출력 폴더에서 JSON 파일 목록 추출
    json_files = get_json_files(output_dir)
    print(f"총 {len(json_files)}개의 JSON 파일에 대해 매칭되는 이미지를 찾습니다.\n")
    
    matched_count = 0
    unmatched_count = 0
    
    # 3단계: 각 JSON 파일에 대응하는 이미지를 2번 폴더에서 찾아 복사
    for json_file in json_files:
        stem = json_file.stem
        image_path = find_matching_image(stem, image_dir)
        
        if image_path:
            dest_path = output_dir / image_path.name
            
            # 이미 동일한 이름의 파일이 있는지 확인
            if dest_path.exists() and dest_path.stat().st_size == image_path.stat().st_size:
                print(f"건너뛰기: {json_file.name} -> {image_path.name} (이미 존재)")
            else:
                shutil.copy2(image_path, dest_path)
                print(f"매칭 성공: {json_file.name} -> {image_path.name}")
            
            matched_count += 1
        else:
            print(f"매칭 실패: {json_file.name}에 대응하는 이미지를 찾을 수 없습니다.")
            unmatched_count += 1
    
    return copied_count, matched_count, unmatched_count


def main():
    parser = argparse.ArgumentParser(
        description="1번 폴더의 모든 파일과 JSON 파일명과 일치하는 이미지를 출력 폴더에 복사합니다."
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="JSON 파일이 있는 폴더 경로 (1번 폴더)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="이미지 파일이 있는 폴더 경로 (2번 폴더)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="결과를 저장할 출력 폴더 경로 (3번 폴더)"
    )
    
    args = parser.parse_args()
    
    json_dir = Path(args.json_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    try:
        copied, matched, unmatched = match_and_copy_images(json_dir, image_dir, output_dir)
        
        print("\n" + "="*60)
        print(f"처리 완료:")
        print(f"  - 1번 폴더에서 복사된 파일: {copied}개")
        print(f"  - 2번 폴더에서 매칭된 이미지: {matched}개")
        print(f"  - 매칭 실패한 JSON: {unmatched}개")
        print(f"  - 출력 디렉토리: {output_dir.absolute()}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        return 1
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
python match_json_images.py \
    --json-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/batch_val/batch_val_phase2_validation/merged_annotations \
    --image-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/2_val_200/sampled_images \
    --output-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/batch_val/batch_val_phase2_validation/merged_annotations_with_images

"""