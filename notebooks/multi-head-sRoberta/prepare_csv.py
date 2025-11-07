import pandas as pd
from pathlib import Path
import shutil
import sys

def collect_all_transcripts(
    label_file_paths: list, 
    source_base_dir: Path, 
    destination_dir: Path
):
    """
    Finds all transcripts based on label files and copies them to a single flat directory.

    Args:
        label_file_paths (list): A list of Paths to your CSV label files 
                                 (train, dev, test).
        source_base_dir (Path): The root folder containing participant 
                                subfolders (e.g., '301_P', '302_P').
        destination_dir (Path): The single flat folder to copy all 
                                transcripts into.
    """
    
    # 1. Create the destination folder if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {destination_dir}\n")

    # 2. Get all unique participant IDs from all label files
    all_participant_ids = set()
    print("Reading participant IDs from label files...")
    for file_path in label_file_paths:
        if not file_path.exists():
            print(f"Warning: Label file not found, skipping: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            if "Participant_ID" not in df.columns:
                print(f"Warning: 'Participant_ID' column not in {file_path}. Skipping.")
                continue
            # Add all IDs from this file's 'Participant_ID' column to the set
            all_participant_ids.update(df["Participant_ID"].astype(str).tolist())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not all_participant_ids:
        print("Error: No participant IDs were loaded. Exiting.")
        sys.exit(1)

    print(f"Found {len(all_participant_ids)} unique participant IDs across all label files.\n")

    # 3. Loop through IDs, find files, and copy them
    copied_count = 0
    skipped_count = 0

    for participant_id in sorted(list(all_participant_ids)): # Sort for nice console output
        
        # This is the path logic from your script:
        # data_directory / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        source_path = source_base_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        
        # This is the new, flat destination path
        destination_path = destination_dir / f"{participant_id}_TRANSCRIPT.csv"

        if source_path.exists():
            try:
                shutil.copyfile(source_path, destination_path)
                print(f"Copied: {source_path.name} -> {destination_dir.name}/")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {source_path.name}: {e}")
                skipped_count += 1
        else:
            print(f"Skipped (Not Found): {source_path}")
            skipped_count += 1
    
    print("\n--- Copy Complete ---")
    print(f"Successfully copied: {copied_count} files")
    print(f"Skipped (Not Found):  {skipped_count} files")
    print(f"Total participants:  {len(all_participant_ids)}")
    print(f"All transcripts are now in: {destination_dir}")


if __name__ == "__main__":
    print("Starting transcript collection...\n")

    # --- 1. CONFIGURE YOUR PATHS HERE ---

    # The folder with all the participant subfolders (e.g., 301_P, 302_P)
    BASE_PATH = Path("D:/volumes/MACBACKUP/extracted_folders")
    
    # The folder where your label CSVs are stored
    LABEL_FOLDER_PATH = Path("D:/volumes/MACBACKUP/Regression")

    # The new, single, flat folder to put all the transcripts
    # You can change 'all_transcripts' to any name you like
    OUTPUT_FLAT_DIR = Path("D:/volumes/MACBACKUP/transcripts")

    # --- 2. SCRIPT EXECUTION ---

    # List of all your label files
    label_files = [
        LABEL_FOLDER_PATH / "train_split_Depression_AVEC2017.csv",
        LABEL_FOLDER_PATH / "dev_split_Depression_AVEC2017.csv",
        LABEL_FOLDER_PATH / "full_test_split.csv"
    ]

    # Run the function
    collect_all_transcripts(
        label_file_paths=label_files,
        source_base_dir=BASE_PATH,
        destination_dir=OUTPUT_FLAT_DIR
    )

    print("\nScript finished.")