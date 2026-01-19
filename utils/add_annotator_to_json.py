# add_annotator_to_json.py
# shape에서 Infinity 포인트 제거 기능 포함!!
import os
import json
import re
import math


def has_infinity_points(shape):
    """
    Shape의 points에 Infinity 또는 -Infinity가 포함되어 있는지 확인한다.
    
    Args:
        shape: LabelMe format의 shape 딕셔너리
    
    Returns:
        bool: Infinity 포인트가 있으면 True, 없으면 False
    """
    points = shape.get('points', [])
    
    for point in points:
        if len(point) != 2:
            continue
        
        x, y = point
        
        # Infinity 또는 -Infinity 체크
        if math.isinf(x) or math.isinf(y):
            return True
    
    return False


def clean_shapes(shapes):
    """
    Shapes 리스트에서 Infinity 포인트를 가진 shape를 제거한다.
    
    Args:
        shapes: LabelMe format의 shapes 리스트
    
    Returns:
        tuple: (cleaned_shapes, removed_count)
    """
    cleaned = []
    removed_count = 0
    
    for shape in shapes:
        if has_infinity_points(shape):
            removed_count += 1
            label = shape.get('label', 'unknown')
            print(f"    ⚠ Removed shape with Infinity points (label: {label})")
        else:
            cleaned.append(shape)
    
    return cleaned, removed_count


def add_annotator_to_json(folder='.'):
    """
    batch_{number}_{name}_annotation 폴더의 JSON 파일에 annotator 속성을 추가하고,
    Infinity 포인트를 가진 shape를 제거한다.
    
    Args:
        folder: 검색할 폴더 경로 (기본값: 현재 디렉토리)
    """
    pattern = re.compile(r'batch_\w+_(.+)_annotation')
    
    total_files_processed = 0
    total_shapes_removed = 0
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        match = pattern.match(subfolder)
        if not match:
            continue
        
        name = match.group(1)
        
        print(f"\n처리 중: {subfolder} (annotator={name})")
        
        folder_files_processed = 0
        folder_shapes_removed = 0
        
        # 폴더 내 JSON 파일 처리
        for file in os.listdir(subfolder_path):
            if not file.endswith('.json'):
                continue
            
            file_path = os.path.join(subfolder_path, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Annotator 추가
                data['annotator'] = name
                
                # Infinity 포인트를 가진 shape 제거
                original_shapes = data.get('shapes', [])
                cleaned_shapes, removed_count = clean_shapes(original_shapes)
                
                if removed_count > 0:
                    print(f"  {file}: {removed_count}개 shape 제거됨")
                    data['shapes'] = cleaned_shapes
                    folder_shapes_removed += removed_count
                
                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                folder_files_processed += 1
                total_files_processed += 1
                
            except Exception as e:
                print(f"  ✗ 오류 발생 ({file}): {e}")
        
        total_shapes_removed += folder_shapes_removed
        
        print(f"✓ {subfolder}: {folder_files_processed}개 파일 처리 완료, {folder_shapes_removed}개 shape 제거됨")
    
    print(f"\n{'='*60}")
    print(f"전체 처리 완료")
    print(f"  처리된 파일: {total_files_processed}개")
    print(f"  제거된 shape: {total_shapes_removed}개")
    print(f"{'='*60}")


if __name__ == '__main__':
    # add_annotator_to_json()  # 현재 디렉토리
    add_annotator_to_json('/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_result')  # 특정 폴더 지정