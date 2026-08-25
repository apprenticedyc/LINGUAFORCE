$ErrorActionPreference = 'Continue'
& 'D:\Anaconda3\python.exe' 'E:\PythonCode\Paper\experiments\extract_dims_full.py' --input 'E:\PythonCode\Paper\COERCION\inputters\data\train.jsonl' --out 'E:\PythonCode\Paper\experiments\output\train_dims.jsonl' --workers 8 --fuse 12 *> 'E:\PythonCode\Paper\experiments\logs\train_dims.out.log'
