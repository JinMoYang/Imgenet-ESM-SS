# Annotation 수집

## 문제 배포

ImageID,annotator1,annotator2,annotator3,Reviewer 1,Reviewer 2,Reviewer 3
ILSVRC2012_val_00022657.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023031.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023088.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023314.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023359.JPEG,우진,은수,홍준,리안,다연,
ILSVRC2012_val_00023463.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00023617.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00023828.JPEG,우진,은수,리안,홍준,다연,
ILSVRC2012_val_00024165.JPEG,우진,은수,리안,홍준,다연,

--> 다음 구조의 csv파일 필요.

csv파일 넣어서 사진 분류하기. copy_images_by_annotator.py 

csv file structure: ImageID, annotator1, annotator2, annotator3

## protocol

phase1 배포 
-> batch_1st_phase1_result 파일에 (batch_{num_string}_{name}_annotation들 정리)

-> 여기서 앞에 접두사 숫자 붙어있는 경우 제거해줘야함 rename_remove_prefix.py (필요없음)

-> add_annotator_to_json.py 로 infinity polygon을 제거하고, json파일에 어노테이터를 추가한다. # 서버에서 실행시 간혹 infinity polygon이 섞일때가 있어서 한번 돌려주는게 좋음

-> distribute_json.py 돌려서 annotator1, annotator2, annotator3 폴더로 분배

-> code_final_final의 merge_annotations.py: 어노테이션 정합하기

-> google form review 

-> 피드백은 구글 스프레드시트 앱스크립트로 작성했음 (시각화 부분에 설명 있음)



Phase2 배포 준비
-> phase2에서 나온 merge_annotations에 있는 json들을 batch_1st_reviewers.csv에 있는 annotator에 맞게 폴더 생성(보통 5개)
이미지도 json에 맞춰 추가해줌
: distribute_annotations.py
CSV: 스프레드시트에 positive row만 남기기 app script를 돌려서 positive row만 남겨야한다.

-> Phase2 진입.

-> batch_1st_phase2_result 파일에 (batch_{num_string}_{name}_annotation들 정리)

-> 여기서 앞에 접두사 숫자 붙어있는 경우 제거해줘야함 rename_remove_prefix.py

-> add_annotator_to_json.py 로 infinity polygon을 제거하고, json파일에 어노테이션을 추가한다.

-> (batch_1st.csv 기반 annotator1, 2, 3으로 매핑, 각 폴더명 batch_{num_string}_{name}_annotation) 
- 이걸 그냥 중복없이 annotator1 부터 폴더안에 저장시키면 될듯. : distribute_json.py

-> code_final_final: 어노테이션 정합 과정

-> google form review 

-> 피드백은 구글 스프레드시트 앱스크립트로 작성

시각화: 
1. merged_annotations 폴더에 있는 json과 이름이 같은 이미지를 sampled_images에서 찾아 새로운 폴더 merged_annotations_with_images를 생성한다 (# match_json_images.py)
2. visualization_all_test.py 돌린다
3. visualized 폴더에 생긴 이미지들과 merged_annotations에 있는 json들 merged_images_phase# 구글 드라이브에 업로드

        구글 드라이브에서

                original_images : 원본 이미지들

                merged_images : 시각화 이미지 + json 

                위치 메인 디렉토리 
                phase1_외부배치_구글폼, phase2 구글폼을 돌려야 함. Apps script에서 상위 폴더를 넣고, original_images, merged_image_phase1, phase2 넣어둠

                리뷰어 CSV 파일도 넣어줘야 함 - 포맷


## feedback 정합과정

phase1: positive, negative
phase2: positive만 -> (스프레드시트에 positive row만 남기기) 코드 돌려서 positive row만 남긴다.

스프레드시트 생성, 이미지-드라이브 매핑: 스프레트시트를 받아 스프레드시트에 복사하고, 이미지 링크를 추가해준다.

생성된 스프레드시트에 리뷰어당 리뷰 매핑 : 위에서 생성된 스프레드시트와 리뷰어당 리뷰가 나온 폼을 넣으면, 열을 만들어서 추가해준다. (현재 구현은 한 리뷰가 5개열 (첫번째 제외하고 4개열 복사))

reviewer1, 2, 3 정보 압축: 이거 사용해서 한줄로 남긴다.

## 최종 피드백 반영

merged_annotations 폴더에 있는 json과 이름이 같은 이미지를 sampled_images에서 찾아 새로운 폴더 merged_annotations_with_images를 생성한다 (# match_json_images.py)

세부로직 : 1번 폴더에 .json 파일들이 있음
        2번 폴더에 이미지 파일들이 있음
        .json 파일 이름과 동일한 이름의 이미지를 2번 폴더에서 찾아서
        3번 폴더에 매칭되는 이미지들을 복사하거나 이동
        폴더명과 위치를 지정 가능하게


## file structure

batch_1st_reviewers.csv 구조: ImageID, annotator1, annotator2, annotator3, Reviewer1, Reviewer2, Reviewer3

feedback_merge.csv 구조: image_name, annotators(띄어쓰기로 구분)


## 추가 유틸

폴더에 있는 .json파일을 모두 찾아 .json 대신 .JPEG을 붙이고 ImageID만 있는 csv 생성 : json_to_jpeg_csv.py

폴더에 있는 .JPEG만 모두 제거 : remove_jpeg_files.py


## 구글 드라이브에 있어야 하는것

1. 이미지 소스
2. batch_1st_reviewers.csv: ImageID, annotator1, annotator2, annotator3, Reviewer1, Reviewer2, Reviewer3