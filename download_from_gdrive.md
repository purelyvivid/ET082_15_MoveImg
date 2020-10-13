# Download File from Google Drive

### Step 1. 在google drive打開檔案共享權限, 設為每個人不須登入就可以檢視

### Step 2. 記下google drive檔案ID 

在檔案上按下滑鼠右鍵-->共享-->複製連結, 可以看到以下url:
https://drive.google.com/file/d/<gdrive的檔案ID>/view?usp=sharing

### Step 3. 下指令

```
git clone https://github.com/chentinghao/download_google_drive.git
cd download_google_drive
python download_gdrive.py <gdrive的檔案ID> <儲存的檔案路徑名稱>
```

其中, <儲存的檔案路徑名稱>是像這樣的格式: `/path/to/file/myfilename.subfilename`

如果不知道當前路徑為何, 可以下指令:
```
python -c "import os; print(os.getcwd())"
```