# googleform_csv_to_feedback_folders.py

import pandas as pd
import shutil
from pathlib import Path
from typing import Set, Dict, List


def parse_csv(csv_path: str) -> pd.DataFrame:
    """
    CSV 파일을 읽고 DataFrame으로 반환한다.
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        pd.DataFrame: CSV 데이터
    """
    return pd.read_csv(csv_path)


def extract_annotators(df: pd.DataFrame, status_column: str = "적절/수정필요") -> Set[str]:
    """
    수정필요 상태의 행들에서 모든 고유한 annotator 이름을 추출한다.
    
    Args:
        df: 원본 DataFrame
        status_column: 상태를 나타내는 열 이름
        
    Returns:
        Set[str]: 고유한 annotator 이름들의 집합
    """
    needs_revision = df[df[status_column] == "수정필요"]
    annotators = set()
    
    for annotator_names in needs_revision["annotators"]:
        if pd.notna(annotator_names):
            names = annotator_names.split()
            annotators.update(names)
    
    return annotators


def create_feedback_folders(annotators: Set[str], base_dir: Path = Path(".")) -> Dict[str, Path]:
    """
    각 annotator별로 피드백 폴더를 생성한다.
    
    Args:
        annotators: annotator 이름들의 집합
        base_dir: 기본 디렉토리 경로
        
    Returns:
        Dict[str, Path]: annotator 이름을 키로, 폴더 경로를 값으로 하는 딕셔너리
    """
    folders = {}
    
    for annotator in annotators:
        folder_path = base_dir / f"feedback_{annotator}"
        folder_path.mkdir(parents=True, exist_ok=True)
        folders[annotator] = folder_path
        print(f"폴더를 생성했습니다: {folder_path}")
    
    return folders


def distribute_images(
    df: pd.DataFrame,
    annotators: Set[str],
    folders: Dict[str, Path],
    source_image_dir: Path,
    status_column: str = "적절/수정필요"
) -> Dict[str, List[int]]:
    """
    수정필요 이미지들을 각 annotator 폴더로 복사한다.
    
    Args:
        df: 원본 DataFrame
        annotators: annotator 이름들의 집합
        folders: annotator별 폴더 경로 딕셔너리
        source_image_dir: 원본 이미지가 있는 디렉토리
        status_column: 상태를 나타내는 열 이름
        
    Returns:
        Dict[str, List[int]]: annotator별로 해당하는 행 인덱스 리스트
    """
    needs_revision = df[df[status_column] == "수정필요"]
    annotator_rows = {annotator: [] for annotator in annotators}
    
    for idx, row in needs_revision.iterrows():
        image_name = row["image_name"]
        annotator_names = row["annotators"]
        
        if pd.notna(annotator_names):
            annotator_list = annotator_names.split()
            
            for annotator in annotator_list:
                if annotator in annotators:
                    # 이미지 복사
                    source_path = source_image_dir / image_name
                    dest_path = folders[annotator] / image_name
                    
                    if source_path.exists():
                        shutil.copy2(source_path, dest_path)
                        print(f"이미지를 복사했습니다: {image_name} -> {folders[annotator]}")
                    else:
                        print(f"경고: 이미지를 찾을 수 없습니다: {source_path}")
                    
                    # 행 인덱스 기록
                    annotator_rows[annotator].append(idx)
    
    return annotator_rows


def create_annotator_csvs(
    df: pd.DataFrame,
    annotator_rows: Dict[str, List[int]],
    folders: Dict[str, Path]
) -> None:
    """
    각 annotator별로 해당하는 행들만 포함하는 CSV 파일을 생성한다.
    
    Args:
        df: 원본 DataFrame
        annotator_rows: annotator별 행 인덱스 리스트
        folders: annotator별 폴더 경로 딕셔너리
    """
    for annotator, row_indices in annotator_rows.items():
        if row_indices:
            annotator_df = df.loc[row_indices]
            csv_path = folders[annotator] / f"feedback_{annotator}.csv"
            annotator_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"CSV 파일을 생성했습니다: {csv_path} ({len(row_indices)}개 행)")


def main(
    csv_path: str,
    source_image_dir: str,
    base_dir: str = "."
) -> None:
    """
    메인 실행 함수.
    
    Args:
        csv_path: 입력 CSV 파일 경로
        source_image_dir: 원본 이미지 디렉토리 경로
        base_dir: 피드백 폴더를 생성할 기본 디렉토리
    """
    # CSV 읽기
    df = parse_csv(csv_path)
    print(f"CSV 파일을 읽었습니다: {len(df)}개 행")
    
    # Annotator 추출
    annotators = extract_annotators(df)
    print(f"추출된 annotator: {', '.join(sorted(annotators))}")
    
    # 폴더 생성
    base_path = Path(base_dir)
    folders = create_feedback_folders(annotators, base_path)
    
    # 이미지 분배 및 행 인덱스 수집
    source_path = Path(source_image_dir)
    annotator_rows = distribute_images(df, annotators, folders, source_path)
    
    # CSV 분할
    create_annotator_csvs(df, annotator_rows, folders)
    
    print("처리가 완료되었습니다.")


if __name__ == "__main__":

    main(
        csv_path="/Users/woojin/Documents/AioT/test/sample_images/feedback_merge.csv",
        source_image_dir="/Users/woojin/Documents/AioT/test/sample_images/sampled_images",
        base_dir="/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation1_feedback"
    )