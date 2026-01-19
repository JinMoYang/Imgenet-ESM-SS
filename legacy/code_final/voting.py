# voting.py
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from polygon_utils import polygons_to_mask, mask_to_polygons

def vote_label(labels: List[str]) -> str:
    """
    여러 레이블 중 다수결로 최종 레이블을 선택한다.
    
    Args:
        labels: 레이블 리스트
    
    Returns:
        가장 많이 등장한 레이블
    """
    if not labels:
        return ""
    
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

def parse_description(description: str) -> List[str]:
    """
    description 문자열을 파싱하여 속성 리스트를 추출한다.
    
    예: "ishole, iscrowd=1" -> ["ishole", "iscrowd=1"]
    
    Args:
        description: description 문자열
    
    Returns:
        속성 리스트
    """
    if not description:
        return []
    
    # 공백, 콤마, "or"로 구분
    description = description.replace(' or ', ',').replace('or', ',')
    attrs = [attr.strip() for attr in re.split(r'[,\s]+', description) if attr.strip()]
    
    return attrs

def vote_description(descriptions: List[str]) -> str:
    """
    여러 description을 다수결로 통합한다.
    
    Args:
        descriptions: description 문자열 리스트
    
    Returns:
        투표로 결정된 description
    """
    if not descriptions:
        return ""
    
    # 모든 description을 속성 단위로 분해
    all_attrs = []
    for desc in descriptions:
        all_attrs.extend(parse_description(desc))
    
    if not all_attrs:
        return ""
    
    # 각 속성별로 카운트
    counter = Counter(all_attrs)
    
    # 과반수 이상 등장한 속성만 포함
    threshold = len(descriptions) / 2
    voted_attrs = [attr for attr, count in counter.items() if count >= threshold]
    
    return ", ".join(sorted(voted_attrs))

def pixel_wise_voting(polygon_groups: Dict[int, List[Tuple[int, int, Any]]], 
                      image_height: int, image_width: int,
                      min_votes: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    pixel 단위 majority voting으로 새로운 폴리곤을 생성한다.
    
    Args:
        polygon_groups: key -> [(annotator_idx, poly_idx, polygon_data), ...] 매핑
        image_height: 이미지 높이
        image_width: 이미지 너비
        min_votes: 최소 필요 투표 수
    
    Returns:
        (승인된 폴리곤 리스트, 기각된 폴리곤 리스트)
    """
    approved = []
    rejected = []
    
    for key, poly_list in polygon_groups.items():
        vote_count = len(poly_list)
        
        if vote_count < min_votes:
            # 투표 미달
            for annotator_idx, poly_idx, poly_data in poly_list:
                rejected.append({
                    'key': key,
                    'votes': vote_count,
                    'data': poly_data
                })
            continue
        
        # 레이블 투표
        labels = [poly_data['label'] for _, _, poly_data in poly_list]
        final_label = vote_label(labels)
        
        # description 투표
        descriptions = [poly_data.get('description', '') for _, _, poly_data in poly_list]
        final_description = vote_description(descriptions)
        
        # pixel 단위 투표를 위한 마스크 생성
        vote_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for _, _, poly_data in poly_list:
            # 각 annotator의 폴리곤들을 마스크로 변환
            mask = polygons_to_mask(poly_data['polygons'], image_height, image_width)
            vote_mask += mask
        
        # min_votes 이상 받은 픽셀만 선택
        final_mask = (vote_mask >= min_votes).astype(np.uint8)
        
        # 마스크가 비어있으면 기각
        if final_mask.sum() == 0:
            for annotator_idx, poly_idx, poly_data in poly_list:
                rejected.append({
                    'key': key,
                    'votes': vote_count,
                    'data': poly_data
                })
            continue
        
        # 마스크에서 새로운 폴리곤 추출
        new_polygons = mask_to_polygons(final_mask)
        
        if not new_polygons:
            # 폴리곤 추출 실패
            for annotator_idx, poly_idx, poly_data in poly_list:
                rejected.append({
                    'key': key,
                    'votes': vote_count,
                    'data': poly_data
                })
            continue
        
        # 승인된 객체 정보 생성
        approved.append({
            'key': key,
            'votes': vote_count,
            'label': final_label,
            'description': final_description,
            'new_polygons': new_polygons,
            'annotators': [annotator_idx for annotator_idx, _, _ in poly_list]
        })
    
    return approved, rejected