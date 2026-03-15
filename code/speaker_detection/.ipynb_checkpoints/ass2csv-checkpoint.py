import json
import os
import pandas as pd 
from datetime import datetime
import pysrt
import re
import csv
import chardet
def zimu_analysis(assfile: str):
    """
    Function: Parse .ass subtitle files.

    Parameters:
    assfile: Subtitle file to be parsed (ass or srt format)
    Returns: None
"""
    if assfile.endswith('ass'):
        ass_result = []
        for encoding in ['utf-8', 'utf-16', 'gbk']:
            try:
                with open(assfile, 'r', encoding=encoding) as file:
                    lines = file.readlines()
                print(f"Successfully read the file with {encoding} encoding.")
                break  
            except UnicodeDecodeError:
                print(f"Failed to read the file with {encoding} encoding. Trying next encoding.")

        for line in lines:
                if line.startswith('Dialogue:'):
                    parts = line.split(',', 9) 
                    if '♪' not in parts[9] and '特效' not in parts[3] and '注释' not in parts[3]: 
                        text = parts[9].replace('\n','').replace('\\N',' ').replace('-','')
                        text = text.lstrip().split(' {')[0]
                        # if text.startswith('{\\fs'):
                        if text.startswith('{\\'):
                            continue
                        text = re.sub(r'{[^}]*}', '', text)

                        start_time_str = datetime.strptime(parts[1], '%H:%M:%S.%f')
                        start_milliseconds = (start_time_str.hour * 3600 + start_time_str.minute * 60 + start_time_str.second) * 1000 + int(start_time_str.microsecond / 1000)
                        start_seconds = start_milliseconds/1000
                        end_time_str = datetime.strptime(parts[2], '%H:%M:%S.%f')
                        end_milliseconds = (end_time_str.hour * 3600 + end_time_str.minute * 60 + end_time_str.second) * 1000 + int(end_time_str.microsecond / 1000)
                        end_seconds = end_milliseconds/1000
                        ass_result.append((start_milliseconds,end_milliseconds,text,text))
    
        res = pd.DataFrame(columns=['start_time','end_time','text','zh_text'])
        res['start_time'],res['end_time'],res['text'],res['zh_text'] = zip(*ass_result)
        csv_file_path = assfile.replace('ass','csv')
        res.to_csv(csv_file_path, index=False,encoding="utf-8-sig")
        return csv_file_path

    if assfile.endswith('.srt'):
        ass_result = []
        
        try:
            with open(assfile, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(assfile, 'r', encoding='gbk') as file:
                lines = file.readlines()


        index = 0
        while index < len(lines):
            # Read and ignore the index line
            index += 1
            
            # Read start and end time line
            time_line = lines[index].strip()
            start_time_str, end_time_str = time_line.split(' --> ')
            
            # Convert timestamps to milliseconds
            start_hours, start_minutes, start_seconds, start_milliseconds = map(int, start_time_str.replace(',', ':').split(':'))
            start_time_in_ms = (start_hours * 3600 + start_minutes * 60 + start_seconds) * 1000 + start_milliseconds
            
            end_hours, end_minutes, end_seconds, end_milliseconds = map(int, end_time_str.replace(',', ':').split(':'))
            end_time_in_ms = (end_hours * 3600 + end_minutes * 60 + end_seconds) * 1000 + end_milliseconds
            
            index += 1
            
            # Read and concatenate subtitle text lines
            text_lines = []
            while index < len(lines) and lines[index].strip() != "":
                text_lines.append(lines[index].strip())
                index += 1
            index += 1  
            full_text = " ".join(text_lines)

            if '♪' not in full_text:
                zh_text = full_text.replace('‎','')
                zh_text = re.sub(r'{[^}]*}', '', zh_text)                    
                zh_text = re.sub(r'[\(\（].*?[\)\）]', '', zh_text)

                if len(zh_text)>0:
                    ass_result.append((start_time_in_ms, end_time_in_ms, zh_text, zh_text))

        # Convert ass_result to DataFrame and save to CSV
        res = pd.DataFrame(columns=['start_time', 'end_time', 'text', 'zh_text'])
        res['start_time'], res['end_time'], res['text'], res['zh_text'] = zip(*ass_result)
        csv_file_path = assfile.replace('.srt', '.csv')
        res.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

        return csv_file_path
    
