import gdown


file_id = '1AQmGqxsMavWbd0fkphHRNr-nXic9wcj8'

url = f'https://drive.google.com/uc?id={file_id}'

output_path = 'downloaded_file.zip'  

gdown.download(url, output_path, quiet=True)  

print("File downloaded successfully.")
