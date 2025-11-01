# save_files.py

import os
import csv

def save_files_to_csv(dir_path, output_csv):
    files = sorted([name for name in os.listdir(dir_path) 
                    if os.path.isfile(os.path.join(dir_path, name))])
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name'])
        for file in files:
            writer.writerow([file])

if __name__ == '__main__':
    save_files_to_csv('/Users/woojin/Documents/AioT/test/sample_images/sampled_images', 'files.csv')
    print(f"저장 완료: files.csv")