# -*- coding: utf-8 -*-
import pandas as pd
import os
import shutil
from datetime import datetime
import os
def read_csv(file_path, encoding="cp1251", sep=";"):
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=sep,
        skipinitialspace=True,
        engine='python'
    )
    return df
def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df

def check_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Предупреждение: путь не существует: {path}")
            return False
    return True

def create_backup():
    source_path = "../rubert_4class_enhanced_v2"

    if not os.path.exists(source_path):
        print("ERROR: Папка с моделью не найдена!")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"../rubert_backup_rubert_4class_enhanced_v2"

    print(f"CREATING BACKUP FROM: {source_path}")
    print(f"SAVING TO: {backup_path}")

    shutil.copytree(source_path, backup_path)
    print(f"SUCCESS: Backup created: {backup_path}")
    print("YOUR MODEL IS NOW SAFE!")

create_backup()

