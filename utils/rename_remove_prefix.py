# rename_remove_prefix.py

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any


def find_files_with_number_prefix(directory: str) -> List[Path]:
    """
    디렉토리 내에서 숫자 접두사를 가진 파일들을 찾습니다.
    
    Args:
        directory: 검색할 디렉토리 경로
    
    Returns:
        숫자 접두사를 가진 파일의 Path 객체 리스트
    """
    pattern = re.compile(r'^\d+[-_]')
    target_dir = Path(directory)
    
    if not target_dir.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    matching_files = []
    for item in target_dir.iterdir():
        if item.is_file() and pattern.match(item.name):
            matching_files.append(item)
    
    return matching_files


def find_json_files(directory: str) -> List[Path]:
    """
    디렉토리 내 모든 JSON 파일을 찾습니다.
    
    Args:
        directory: 검색할 디렉토리 경로
    
    Returns:
        JSON 파일의 Path 객체 리스트
    """
    target_dir = Path(directory)
    
    if not target_dir.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    return list(target_dir.glob("*.json"))


def remove_number_prefix(text: str) -> str:
    """
    텍스트에서 숫자 접두사를 제거합니다.
    
    Args:
        text: 원본 텍스트
    
    Returns:
        접두사가 제거된 텍스트
    """
    pattern = re.compile(r'^\d+[-_]')
    return pattern.sub('', text)


def generate_new_name(old_name: str) -> str:
    """
    파일명에서 숫자 접두사를 제거합니다.
    
    Args:
        old_name: 원본 파일명
    
    Returns:
        접두사가 제거된 새 파일명
    """
    return remove_number_prefix(old_name)


def check_conflicts(rename_pairs: List[Tuple[Path, str]], target_dir: Path) -> List[str]:
    """
    이름 변경 시 충돌이 발생하는지 확인합니다.
    
    Args:
        rename_pairs: (원본 Path, 새 이름) 튜플 리스트
        target_dir: 대상 디렉토리
    
    Returns:
        충돌하는 파일명 리스트
    """
    conflicts = []
    new_names = set()
    
    for old_path, new_name in rename_pairs:
        new_path = target_dir / new_name
        if new_path.exists() and new_path != old_path:
            conflicts.append(new_name)
        if new_name in new_names:
            conflicts.append(f"{new_name} (중복)")
        new_names.add(new_name)
    
    return conflicts


def process_json_file(json_path: Path, backup: bool = True) -> Tuple[bool, str]:
    """
    JSON 파일의 imagePath 필드에서 숫자 접두사를 제거합니다.
    
    Args:
        json_path: JSON 파일 경로
        backup: 백업 파일 생성 여부
    
    Returns:
        (성공 여부, 메시지) 튜플
    """
    try:
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # imagePath 필드 확인 및 처리
        if 'imagePath' not in data:
            return False, "imagePath 필드 없음"
        
        old_image_path = data['imagePath']
        new_image_path = remove_number_prefix(old_image_path)
        
        # 변경사항이 없으면 스킵
        if old_image_path == new_image_path:
            return False, "변경 불필요"
        
        # 백업 생성
        if backup:
            backup_path = json_path.with_suffix('.json.bak')
            shutil.copy2(json_path, backup_path)
        
        # imagePath 업데이트
        data['imagePath'] = new_image_path
        
        # JSON 파일 저장 (원본 포맷 유지)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, f"{old_image_path} → {new_image_path}"
    
    except json.JSONDecodeError as e:
        return False, f"JSON 파싱 오류: {e}"
    except Exception as e:
        return False, f"처리 오류: {e}"


def update_json_files(directory: str, dry_run: bool = False, backup: bool = True) -> None:
    """
    디렉토리 내 모든 JSON 파일의 imagePath를 업데이트합니다.
    
    Args:
        directory: 대상 디렉토리 경로
        dry_run: True인 경우 실제 변경 없이 미리보기만 수행
        backup: 백업 파일 생성 여부
    """
    json_files = find_json_files(directory)
    
    if not json_files:
        print("JSON 파일을 찾지 못했습니다.")
        return
    
    print(f"\n총 {len(json_files)}개의 JSON 파일을 발견했습니다.\n")
    
    # 변경 내역 수집
    changes = []
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'imagePath' in data:
                old_path = data['imagePath']
                new_path = remove_number_prefix(old_path)
                if old_path != new_path:
                    changes.append((json_path, old_path, new_path))
        except Exception as e:
            print(f"오류 ({json_path.name}): {e}")
    
    if not changes:
        print("변경이 필요한 파일이 없습니다.")
        return
    
    print(f"변경 예정 ({len(changes)}개 파일):")
    for json_path, old_path, new_path in changes:
        print(f"  [{json_path.name}] {old_path} → {new_path}")
    
    if dry_run:
        print("\n[Dry Run] 실제 변경은 수행되지 않았습니다.")
        return
    
    # 사용자 확인
    print("\n계속 진행하시겠습니까? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("작업이 취소되었습니다.")
        return
    
    # 실제 처리
    success_count = 0
    for json_path, _, _ in changes:
        success, message = process_json_file(json_path, backup=backup)
        if success:
            success_count += 1
            print(f"완료: {json_path.name}")
        else:
            print(f"스킵: {json_path.name} ({message})")
    
    print(f"\n완료: {success_count}/{len(changes)}개의 JSON 파일이 업데이트되었습니다.")
    if backup:
        print("백업 파일이 .bak 확장자로 저장되었습니다.")


def rename_files(directory: str, dry_run: bool = False) -> None:
    """
    디렉토리 내 파일명에서 숫자 접두사를 제거합니다.
    
    Args:
        directory: 대상 디렉토리 경로
        dry_run: True인 경우 실제 변경 없이 미리보기만 수행
    """
    target_dir = Path(directory)
    files = find_files_with_number_prefix(directory)
    
    if not files:
        print("숫자 접두사를 가진 파일을 찾지 못했습니다.")
        return
    
    print(f"\n총 {len(files)}개의 파일을 발견했습니다.\n")
    
    # 변경 계획 생성
    rename_pairs = [(f, generate_new_name(f.name)) for f in files]
    
    # 충돌 확인
    conflicts = check_conflicts(rename_pairs, target_dir)
    if conflicts:
        print("다음 파일명과 충돌이 발생합니다:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        return
    
    # 변경 내역 출력
    print("변경 예정:")
    for old_path, new_name in rename_pairs:
        print(f"  {old_path.name} → {new_name}")
    
    if dry_run:
        print("\n[Dry Run] 실제 변경은 수행되지 않았습니다.")
        return
    
    # 사용자 확인
    print("\n계속 진행하시겠습니까? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("작업이 취소되었습니다.")
        return
    
    # 실제 이름 변경
    success_count = 0
    for old_path, new_name in rename_pairs:
        try:
            new_path = target_dir / new_name
            old_path.rename(new_path)
            success_count += 1
        except Exception as e:
            print(f"오류 발생 ({old_path.name}): {e}")
    
    print(f"\n완료: {success_count}/{len(files)}개의 파일명이 변경되었습니다.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="파일명 및 JSON의 imagePath에서 숫자 접두사를 제거합니다."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="대상 디렉토리 (기본값: 현재 디렉토리)"
    )
    parser.add_argument(
        "--mode",
        choices=["files", "json", "both"],
        default="both",
        help="처리 모드: files(파일명만), json(JSON만), both(둘 다, 기본값)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 변경 없이 미리보기만 수행합니다"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="JSON 파일 백업을 생성하지 않습니다"
    )
    
    args = parser.parse_args()
    
    if args.mode in ["files", "both"]:
        print("=" * 50)
        print("파일명 변경 작업")
        print("=" * 50)
        rename_files(args.directory, dry_run=args.dry_run)
    
    if args.mode in ["json", "both"]:
        print("\n" + "=" * 50)
        print("JSON 파일 업데이트 작업")
        print("=" * 50)
        update_json_files(
            args.directory, 
            dry_run=args.dry_run, 
            backup=not args.no_backup
        )


if __name__ == "__main__":
    main()


"""
# 파일명과 JSON 모두 처리 (미리보기)
python rename_remove_prefix.py --dry-run

# 파일명만 처리
python rename_remove_prefix.py --mode files

# JSON만 처리
python rename_remove_prefix.py --mode json

# 특정 디렉토리 지정
python rename_remove_prefix.py /path/to/folder

# 백업 없이 JSON 처리
python rename_remove_prefix.py --mode json --no-backup

# 전체 실행 (파일명 + JSON)
python rename_remove_prefix.py

"""

# python rename_remove_prefix.py /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_2nd/batch_2nd_answer/batch_2nd_은수_annotation --mode json

# python rename_remove_prefix.py /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/batch_test/batch_test_result/batch_test_지예준_annotation