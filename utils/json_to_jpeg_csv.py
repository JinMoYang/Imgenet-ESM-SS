# scripts/json_to_jpeg_csv.py

import csv
from pathlib import Path


def create_image_id_csv(folder_path, output_csv_path, case_sensitive=True):
    """
    폴더 내 모든 .json 파일을 찾아 .JPEG 확장자로 변경한 ImageID CSV를 생성합니다.
    
    Args:
        folder_path: 검색할 폴더 경로
        output_csv_path: 출력 CSV 파일 경로
        case_sensitive: 대소문자를 구분하여 정렬할지 여부 (기본값: True)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
    
    # .json 파일 수집 및 변환
    image_ids = []
    for json_file in folder.glob("*.json"):
        # 확장자 제거 후 .JPEG 추가
        image_id = json_file.stem + ".JPEG"
        image_ids.append(image_id)
    
    # 정렬
    if case_sensitive:
        image_ids = sorted(image_ids)
    else:
        image_ids = sorted(image_ids, key=str.lower)
    
    # CSV 작성
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID"])
        for image_id in image_ids:
            writer.writerow([image_id])
    
    print(f"총 {len(image_ids)}개의 ImageID를 정렬하여 {output_csv_path}에 저장했습니다.")


def main():
    folder_path = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_validation/merged_annotations"  # 탐색할 폴더 경로
    output_csv_path = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_question/batch_3rd_phase2.csv"  # 출력 CSV 경로
    
    create_image_id_csv(folder_path, output_csv_path, case_sensitive=False)


if __name__ == "__main__":
    main()