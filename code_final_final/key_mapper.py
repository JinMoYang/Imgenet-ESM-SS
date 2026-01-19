# key_mapper.py
from typing import Dict, Tuple, Optional

class KeyMapper:
    """
    폴리곤 간 key mapping을 관리하는 클래스.
    IoU가 threshold를 넘는 폴리곤들에게 동일한 key를 할당한다.
    """
    
    def __init__(self):
        self.poly_to_key: Dict[Tuple[int, int], int] = {}  # (annotator_idx, poly_idx) -> key
        self.next_key = 1
    
    def get_key(self, annotator_idx: int, poly_idx: int) -> Optional[int]:
        """
        특정 폴리곤의 할당된 key를 반환한다.
        
        Args:
            annotator_idx: annotator 인덱스
            poly_idx: 폴리곤 인덱스
        
        Returns:
            할당된 key, 없으면 None
        """
        return self.poly_to_key.get((annotator_idx, poly_idx))
    
    def update(self, annotator_idx1: int, poly_idx1: int, 
               annotator_idx2: int, poly_idx2: int) -> int:
        """
        두 폴리곤에 동일한 key를 할당한다.
        
        Args:
            annotator_idx1: 첫 번째 annotator 인덱스
            poly_idx1: 첫 번째 폴리곤 인덱스
            annotator_idx2: 두 번째 annotator 인덱스
            poly_idx2: 두 번째 폴리곤 인덱스
        
        Returns:
            할당된 key
        """
        key1 = self.get_key(annotator_idx1, poly_idx1)
        key2 = self.get_key(annotator_idx2, poly_idx2)
        
        if key1 is not None and key2 is not None:
            # 둘 다 이미 key가 있는 경우, 더 작은 key로 통일
            target_key = min(key1, key2)
            self._merge_keys(key1, key2, target_key)
            return target_key
        elif key1 is not None:
            # poly1만 key가 있는 경우
            self.poly_to_key[(annotator_idx2, poly_idx2)] = key1
            return key1
        elif key2 is not None:
            # poly2만 key가 있는 경우
            self.poly_to_key[(annotator_idx1, poly_idx1)] = key2
            return key2
        else:
            # 둘 다 key가 없는 경우, 새 key 할당
            new_key = self.next_key
            self.next_key += 1
            self.poly_to_key[(annotator_idx1, poly_idx1)] = new_key
            self.poly_to_key[(annotator_idx2, poly_idx2)] = new_key
            return new_key
    
    def _merge_keys(self, key1: int, key2: int, target_key: int):
        """
        두 개의 서로 다른 key를 하나로 통합한다.
        """
        for poly_id, key in list(self.poly_to_key.items()):
            if key == key1 or key == key2:
                self.poly_to_key[poly_id] = target_key
    
    def get_groups(self) -> Dict[int, list]:
        """
        key별로 폴리곤들을 그룹화한 결과를 반환한다.
        
        Returns:
            key -> [(annotator_idx, poly_idx), ...] 매핑
        """
        groups = {}
        for poly_id, key in self.poly_to_key.items():
            if key not in groups:
                groups[key] = []
            groups[key].append(poly_id)
        return groups