# polygon_utils.py
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Dict, Any

def compute_iou(poly1_points: List[List[float]], poly2_points: List[List[float]]) -> float:
    """
    두 폴리곤의 IoU를 계산한다.
    
    Args:
        poly1_points: 첫 번째 폴리곤의 좌표 리스트
        poly2_points: 두 번째 폴리곤의 좌표 리스트
    
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    try:
        p1 = Polygon(poly1_points)
        p2 = Polygon(poly2_points)
        
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception as e:
        print(f"IoU 계산 중 오류 발생: {e}")
        return 0.0

def merge_polygon_group(polygons: List[Dict[str, Any]]) -> List[List[float]]:
    """
    group_id가 동일한 폴리곤들을 하나의 좌표 리스트로 병합한다.
    
    Args:
        polygons: 동일 group_id를 가진 폴리곤 리스트
    
    Returns:
        병합된 폴리곤의 좌표 리스트
    """
    if len(polygons) == 1:
        return polygons[0]['points']
    
    try:
        shapely_polygons = [Polygon(p['points']) for p in polygons]
        merged = unary_union(shapely_polygons)
        
        if merged.geom_type == 'Polygon':
            return list(merged.exterior.coords[:-1])
        elif merged.geom_type == 'MultiPolygon':
            # MultiPolygon인 경우 가장 큰 폴리곤 반환
            largest = max(merged.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords[:-1])
    except Exception as e:
        print(f"폴리곤 병합 중 오류 발생: {e}")
        return polygons[0]['points']

def group_polygons_by_group_id(shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    group_id를 기준으로 폴리곤들을 그룹화한다.
    
    Args:
        shapes: JSON의 shapes 리스트
    
    Returns:
        그룹화된 폴리곤 리스트 (각 항목은 하나 이상의 폴리곤을 포함)
    """
    groups = {}
    next_group_id = 1
    
    for shape in shapes:
        gid = shape.get('group_id')
        
        if gid is None:
            # group_id가 없으면 개별 객체로 취급
            gid = f"auto_{next_group_id}"
            next_group_id += 1
        
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(shape)
    
    # 각 그룹을 단일 표현으로 변환
    result = []
    for gid, shapes_in_group in groups.items():
        result.append({
            'original_group_id': gid,
            'polygons': shapes_in_group,
            'label': shapes_in_group[0]['label'],
            'description': shapes_in_group[0].get('description', ''),
            'merged_points': merge_polygon_group(shapes_in_group)
        })
    
    return result

def polygon_to_mask(points: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    폴리곤 좌표를 binary mask로 변환한다.
    
    Args:
        points: 폴리곤 좌표 리스트 [[x1, y1], [x2, y2], ...]
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        binary mask (uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def polygons_to_mask(polygons: List[Dict[str, Any]], height: int, width: int) -> np.ndarray:
    """
    여러 폴리곤을 하나의 binary mask로 변환한다.
    group_id로 그룹화된 폴리곤들을 모두 포함한다.
    
    Args:
        polygons: 폴리곤 리스트 (group_id로 묶인 조각들 포함)
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        binary mask (uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for poly in polygons:
        pts = np.array(poly['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    
    return mask

def mask_to_polygons(mask: np.ndarray, epsilon_factor: float = 0.001) -> List[List[List[float]]]:
    """
    binary mask에서 폴리곤 contour를 추출한다.
    
    Args:
        mask: binary mask (0 또는 1)
        epsilon_factor: contour 단순화 계수 (perimeter 대비 비율)
    
    Returns:
        폴리곤 좌표 리스트의 리스트 [[[x1, y1], [x2, y2], ...], ...]
    """
    # uint8로 변환
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # contour 찾기
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # 너무 작은 contour는 제외
        if cv2.contourArea(contour) < 10:
            continue
        
        # contour 단순화
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 3개 미만의 점은 폴리곤이 아님
        if len(approx) < 3:
            continue
        
        # [[x, y], [x, y], ...] 형태로 변환
        points = approx.reshape(-1, 2).astype(float).tolist()
        polygons.append(points)
    
    return polygons