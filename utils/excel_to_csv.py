# excel_to_csv.py

import pandas as pd

def excel_to_csv(excel_path, csv_path):
    """
    Excel 파일을 CSV 파일로 변환한다.
    
    Args:
        excel_path: 입력 Excel 파일 경로
        csv_path: 출력 CSV 파일 경로
    """
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    excel_to_csv('files.xlsx', 'batch_1st.csv')
    print(f"변환 완료: output.csv")