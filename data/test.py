# import requests
# import os
# import zipfile
# from tqdm import tqdm

# # --- Configuration ---
# base_url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
# file_numbers = range(449, 493) 

# # Set this to the full path of your hard drive folder
# output_folder = '/Volumes/MACBACKUP/extracted_folders'
# # --- End of Configuration ---

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#     print(f"Created main directory on your hard drive: {output_folder}")

# for number in file_numbers:
#     file_name = f"{number}_P.zip"
#     file_url = f"{base_url}{file_name}"
#     zip_path = os.path.join(output_folder, file_name)
    
#     print(f"\n--- Processing folder #{number} ---")
    
#     try:
#         response = requests.get(file_url, stream=True)
        
#         if response.status_code == 200:
#             # --- 1. DOWNLOAD THE ZIP FILE ---
#             total_size = int(response.headers.get('content-length', 0))
#             with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name) as progress_bar:
#                 with open(zip_path, 'wb') as f:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         f.write(chunk)
#                         progress_bar.update(len(chunk))
            
#             try:
#                 # --- 2. UNZIP THE FILE ---
#                 extract_path = os.path.join(output_folder, file_name[:-4])
#                 print(f"Unzipping to {extract_path}...")
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     zip_ref.extractall(extract_path)
#                 print(f"Successfully extracted.")
                
#                 # --- 3. CLEAN THE .TXT AND .BIN FILES ---
#                 print("Cleaning .txt and .bin files from new folder...")
#                 cleaned_count = 0
#                 for root, dirs, files in os.walk(extract_path):
#                     for file in files:
#                         # ** THIS IS THE MODIFIED LINE **
#                         if file.endswith(('.txt', '.bin')):
#                             file_to_delete = os.path.join(root, file)
#                             os.remove(file_to_delete)
#                             cleaned_count += 1
#                 print(f"Cleaned {cleaned_count} file(s).")

#                 # --- 4. DELETE THE ORIGINAL ZIP FILE ---
#                 os.remove(zip_path)
#                 print(f"Deleted original zip file: {file_name}")

#             except zipfile.BadZipFile:
#                 print(f"Error: {file_name} is not a valid zip file. It will not be deleted.")

#         elif response.status_code == 404:
#             print(f"File not found on server. Skipping.")
#         else:
#             print(f"Failed to download. Status code: {response.status_code}")

#     except requests.exceptions.RequestException as e:
#         print(f"A connection error occurred: {e}")

# print("\n--- All tasks finished! ---")

import os

# 1. Set the full path to the main folder containing all your 'XXX_P' subfolders.
root_folder = '/Volumes/MACBACKUP/extracted_folders'

# --- Safety Check ---
if not os.path.isdir(root_folder):
    print(f"Error: The directory '{root_folder}' was not found.")
    print("Please make sure the script is in the correct location and the folder name is correct.")
    exit()

print(f"--- Starting verification for folders in '{root_folder}' ---\n")

folders_with_issues = 0

# Get a list of all items in the root folder that are directories
subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

for folder_name in subfolders:
    # ** THIS IS THE CORRECTED LOGIC **
    # Get the first 3 characters of the folder name to use as the file prefix.
    file_prefix = folder_name[:3]

    # 2. Define the set of 4 files we expect to find based on the prefix.
    expected_files = {
        f"{file_prefix}_COVAREP.csv",
        f"{file_prefix}_FORMANT.csv",
        f"{file_prefix}_TRANSCRIPT.csv",
        f"{file_prefix}_AUDIO.wav"
    }

    folder_path = os.path.join(root_folder, folder_name)
    
    # 3. Get the set of actual files currently in the folder.
    try:
        actual_files = set(os.listdir(folder_path))
    except OSError as e:
        print(f"Could not read folder {folder_name}: {e}")
        continue

    # 4. Compare the sets to find discrepancies.
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    # 5. Report any issues found for the current folder.
    if missing_files or extra_files:
        folders_with_issues += 1
        print(f"Issues found in folder: {folder_name}")
        if missing_files:
            print(f"   - MISSING files: {list(missing_files)}")
        if extra_files:
            print(f"   - EXTRA files:   {list(extra_files)}")
        print("-" * 20)

# --- Final Summary ---
print("\n--- Verification Complete ---")
if folders_with_issues == 0:
    print("All folders have the correct 4 files. No issues found.")
else:
    print(f"Found issues in {folders_with_issues} folder(s). Please review the log above.")