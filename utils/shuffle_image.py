# scripts/shuffle_images.py

import random
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def collect_images(root_dir: Path) -> List[Path]:
    """
    root_dir 하위의 모든 이미지 파일 경로를 수집한다.
    
    Args:
        root_dir: 탐색할 최상위 디렉토리
        
    Returns:
        이미지 파일 경로 리스트
    """
    images = []
    for path in root_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def shuffle_and_rename(images: List[Path], start_index: int = 0) -> None:
    """
    이미지 리스트를 섞은 후 순서대로 번호를 붙여 리네임한다.
    
    Args:
        images: 이미지 파일 경로 리스트
        start_index: 시작 인덱스 (기본값 0)
    """
    if not images:
        print("이미지 파일이 없습니다.")
        return
    
    # 섞기
    random.shuffle(images)
    
    # 충돌 방지를 위한 임시 리네임
    temp_mappings = []
    for i, path in enumerate(images):
        temp_name = f"__temp_{i}_{path.name}"
        temp_path = path.parent / temp_name
        path.rename(temp_path)
        temp_mappings.append((temp_path, path.name, i))
    
    # 최종 리네임
    for temp_path, original_name, idx in temp_mappings:
        final_name = f"{start_index + idx}-{original_name}"
        final_path = temp_path.parent / final_name
        temp_path.rename(final_path)
        print(f"리네임 완료: {final_path}")


def main():
    """
    메인 실행 함수.
    사용자로부터 폴더 경로를 입력받아 이미지를 섞는다.
    """
    root_input = input("이미지를 섞을 상위 폴더 경로를 입력하세요: ").strip()
    root_dir = Path(root_input)
    
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"오류: '{root_input}'는 유효한 폴더가 아닙니다.")
        return
    
    print(f"'{root_dir}' 하위의 이미지를 수집 중입니다...")
    images = collect_images(root_dir)
    print(f"총 {len(images)}개의 이미지를 찾았습니다.")
    
    if images:
        confirm = input("이미지를 섞고 리네임하시겠습니까? (y/n): ").strip().lower()
        if confirm == 'y':
            shuffle_and_rename(images)
            print("모든 이미지 섞기가 완료되었습니다.")
        else:
            print("작업이 취소되었습니다.")


if __name__ == "__main__":
    main()