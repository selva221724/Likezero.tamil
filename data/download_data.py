import gdown
import os

def download_file_from_google_drive(file_id, save_path):
    """Download a file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, save_path, quiet=False)

data_path = "data"

# Get the directory of the current script or notebook
current_directory = os.path.abspath(os.getcwd())
# Navigate to the desired directory 
parent_directory = os.path.dirname(current_directory)
os.chdir(parent_directory)
print(f"Current working directory: {os.getcwd()}")

if not os.path.exists(data_path):
    os.makedirs(data_path)
    
# Download the data file
file_id = "1htLDidC0prrYTdef38SJcZj5MykPqnSt"
save_path = os.path.join(data_path, "Fake.csv")

file_id2 = "1yhQQfV0Bxl-m4WSc4VUi13JuZ3FW1pdo"
save_path2 = os.path.join(data_path, "True.csv")

file_id3 = "13cnGEo1dHh4kej29yBhDBhZ8pTiO_tOt"
save_path3 = os.path.join(data_path, "Test.csv")

download_file_from_google_drive(file_id, save_path)
download_file_from_google_drive(file_id2, save_path2)
download_file_from_google_drive(file_id3, save_path3) 
