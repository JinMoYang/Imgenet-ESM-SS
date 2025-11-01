# add_annotator_to_json.py

import os
import json
import re

def add_annotator_to_json(folder='.'):
    """
    batch_{number}_{name}_annotation 폴더의 JSON 파일에 annotator 속성을 추가한다.
    
    Args:
        folder: 검색할 폴더 경로 (기본값: 현재 디렉토리)
    """
    pattern = re.compile(r'batch_\w+_(.+)_annotation')
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        match = pattern.match(subfolder)
        if not match:
            continue
        
        name = match.group(1)
        
        # 폴더 내 JSON 파일 처리
        for file in os.listdir(subfolder_path):
            if not file.endswith('.json'):
                continue
            
            file_path = os.path.join(subfolder_path, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['annotator'] = name
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{subfolder}: JSON 파일에 annotator={name} 추가 완료")

if __name__ == '__main__':
    # add_annotator_to_json()  # 현재 디렉토리
    add_annotator_to_json('/Users/woojin/Documents/AioT/test/sample_images/batch_1st/batch_1st_result')  # 특정 폴더 지정