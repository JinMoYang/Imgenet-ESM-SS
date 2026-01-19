
import argparse
import os
import shutil
from pathlib import Path
from typing import List

def create_output_structure(output_dir: Path, num_annotators: int = 3) -> None:
    """
    출력 디렉토리와 지정된 개수의 annotator 하위 디렉토리를 생성합니다.
    
    Args:
        output_dir: 출력 디렉토리 경로
        num_annotators: 생성할 annotator 디렉토리 개수
    """
    output_dir.mkdir(exist_ok=True)
    
    for i in range(1, num_annotators + 1):
        annotator_dir = output_dir / f"annotator{i}"
        annotator_dir.mkdir(exist_ok=True)
        print(f"디렉토리 생성 완료: {annotator_dir}")

def find_available_annotator(output_dir: Path, filename: str, num_annotators: int = 3) -> Path:
    """
    주어진 파일명이 저장 가능한 annotator 디렉토리를 찾습니다.
    annotator1부터 순서대로 확인하여 중복되지 않는 첫 번째 디렉토리를 반환합니다.
    
    Args:
        output_dir: 출력 디렉토리 경로
        filename: 확인할 파일명
        num_annotators: 확인할 annotator 디렉토리 개수
        
    Returns:
        파일을 저장할 annotator 디렉토리 경로
    """
    for i in range(1, num_annotators + 1):
        annotator_dir = output_dir / f"annotator{i}"
        target_path = annotator_dir / filename
        
        if not target_path.exists():
            return annotator_dir
    
    # 모든 annotator가 찬 경우, 마지막 annotator를 반환 (덮어쓰기)
    return output_dir / f"annotator{num_annotators}"

def distribute_json_files(source_folders: List[str], output_dir: str, num_annotators: int = 3) -> None:
    """
    여러 폴더에서 JSON 파일을 찾아 annotator 폴더들에 분배합니다.
    
    Args:
        source_folders: JSON 파일을 찾을 원본 폴더 경로 리스트
        output_dir: 출력 디렉토리 경로
        num_annotators: 생성할 annotator 디렉토리 개수
    """
    output_path = Path(output_dir)
    
    # 출력 디렉토리 구조 생성
    print("출력 디렉토리 구조를 생성합니다.")
    create_output_structure(output_path, num_annotators)
    print()
    
    # 각 원본 폴더를 순회
    for folder in source_folders:
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"경고: 폴더를 찾을 수 없습니다 - {folder}")
            continue
        
        print(f"폴더 탐색 중: {folder}")
        
        # 폴더 내 모든 JSON 파일 찾기 (정렬)
        json_files = sorted(folder_path.glob("*.json"))
        
        if not json_files:
            print(f"  JSON 파일이 발견되지 않았습니다.")
            continue
        
        # 각 JSON 파일 복사
        for json_file in json_files:
            filename = json_file.name
            target_dir = find_available_annotator(output_path, filename, num_annotators)
            target_path = target_dir / filename
            
            # 파일 복사
            shutil.copy2(json_file, target_path)
            print(f"  {filename} -> {target_dir.name}/")
    
    print()
    print("모든 JSON 파일 분배가 완료되었습니다.")

def get_subdirectories(parent_dir: Path) -> List[str]:
    """
    상위 디렉토리 내의 모든 하위 디렉토리를 찾아 정렬된 리스트로 반환합니다.
    
    Args:
        parent_dir: 상위 디렉토리 경로
        
    Returns:
        하위 디렉토리 경로 리스트 (정렬됨)
    """
    if not parent_dir.exists():
        print(f"오류: 상위 디렉토리를 찾을 수 없습니다 - {parent_dir}")
        return []
    
    if not parent_dir.is_dir():
        print(f"오류: 지정된 경로가 디렉토리가 아닙니다 - {parent_dir}")
        return []
    
    # 하위 디렉토리만 필터링하여 정렬
    subdirs = [str(item) for item in parent_dir.iterdir() if item.is_dir()]
    subdirs.sort()
    
    return subdirs

def main():
    """
    메인 실행 함수
    """
    parser = argparse.ArgumentParser(
        description="여러 폴더에서 JSON 파일을 찾아 annotator 폴더들에 분배합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python distribute_json.py -s /path/to/source_parent -o /path/to/output
  python distribute_json.py --source /data/folders --output /data/annotations
  python distribute_json.py -s ./source_folders -o ./output_annotations --num-annotators 5
        """
    )
    
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="JSON 파일이 있는 하위 폴더들을 포함하는 상위 디렉토리 경로"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="annotator 폴더들을 생성할 출력 디렉토리 경로"
    )
    
    parser.add_argument(
        "-n", "--num-annotators",
        type=int,
        default=3,
        help="생성할 annotator 폴더 개수 (기본값: 3)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 상위 디렉토리에서 하위 폴더 자동 탐색
    parent_dir = Path(args.source).resolve()
    source_folders = get_subdirectories(parent_dir)
    
    if not source_folders:
        print(f"오류: {parent_dir} 내에 하위 폴더가 없습니다.")
        return
    
    if args.verbose:
        print(f"상위 디렉토리: {parent_dir}")
        print(f"발견된 하위 폴더 개수: {len(source_folders)}")
        for folder in source_folders:
            print(f"  - {Path(folder).name}")
        print()
    
    # 출력 디렉토리
    output_dir = args.output
    
    # JSON 파일 분배 실행
    distribute_json_files(source_folders, output_dir, args.num_annotators)
    
    # 결과 확인
    print("\n=== 결과 확인 ===")
    output_path = Path(output_dir)
    for i in range(1, args.num_annotators + 1):
        annotator_dir = output_path / f"annotator{i}"
        if not annotator_dir.exists():
            continue
        files = list(annotator_dir.glob("*.json"))
        print(f"\nannotator{i} ({len(files)}개 파일):")
        for f in sorted(files):
            print(f"  - {f.name}")

if __name__ == "__main__":
    main()

## python distribute_json.py -s /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_result -o /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_validation -v

