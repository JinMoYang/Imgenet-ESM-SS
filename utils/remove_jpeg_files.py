# scripts/remove_jpeg_files.py

from pathlib import Path
from typing import List, Tuple


def remove_jpeg_files(
    directory: str,
    recursive: bool = False,
    dry_run: bool = False
) -> Tuple[List[Path], int]:
    """
    지정된 디렉토리에서 .jpeg 확장자 파일을 삭제한다.
    
    Args:
        directory: 탐색할 디렉토리 경로
        recursive: True일 경우 하위 디렉토리까지 탐색
        dry_run: True일 경우 실제 삭제 없이 대상 파일만 출력
    
    Returns:
        삭제된(또는 삭제 대상) 파일 경로 리스트와 파일 개수
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {directory}")
    
    # .jpeg, .JPEG 파일 탐색
    pattern = "**/*.jpeg" if recursive else "*.jpeg"
    jpeg_files_lower = list(dir_path.glob(pattern))
    
    pattern = "**/*.JPEG" if recursive else "*.JPEG"
    jpeg_files_upper = list(dir_path.glob(pattern))
    
    target_files = jpeg_files_lower + jpeg_files_upper
    
    deleted_files = []
    
    for file_path in target_files:
        try:
            if dry_run:
                print(f"[DRY RUN] 삭제 대상: {file_path}")
                deleted_files.append(file_path)
            else:
                file_path.unlink()
                print(f"삭제 완료: {file_path}")
                deleted_files.append(file_path)
        except PermissionError:
            print(f"권한 오류로 삭제 실패: {file_path}")
        except Exception as e:
            print(f"삭제 중 오류 발생: {file_path}, 오류: {e}")
    
    return deleted_files, len(deleted_files)


def main():
    """실행 예시"""
    # 현재 디렉토리에서 .jpeg 파일 삭제
    target_dir = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_validation/merged_annotations"
    
    # 먼저 dry_run으로 삭제 대상 확인
    print("=== 삭제 대상 파일 확인 ===")
    files, count = remove_jpeg_files(target_dir, recursive=False, dry_run=True)
    print(f"\n총 {count}개의 파일이 삭제 대상입니다.")
    
    # 실제 삭제 실행
    if count > 0:
        confirm = input("\n위 파일들을 삭제하시겠습니까? (y/n): ")
        if confirm.lower() == 'y':
            print("\n=== 파일 삭제 실행 ===")
            files, count = remove_jpeg_files(target_dir, recursive=False, dry_run=False)
            print(f"\n총 {count}개의 파일을 삭제했습니다.")
        else:
            print("삭제가 취소되었습니다.")


if __name__ == "__main__":
    main()