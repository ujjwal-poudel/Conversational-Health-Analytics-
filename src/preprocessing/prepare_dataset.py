import pandas as pd
from pathlib import Path

def process_transcripts(
    data_directory: Path,
    label_df: pd.DataFrame,
    participant_csv: Path = None,
    ellie_csv: Path = None,
    save_files: bool = True
):
    """
    Reads all transcript files, separates participant and Ellie speech,
    and prepares a processed dataset.

    Args:
        data_directory (Path): Path to the folder containing participant data.
        label_df (pd.DataFrame): DataFrame containing Participant_IDs and labels.
        participant_csv (Path, optional): Path to save participant speech CSV.
        ellie_csv (Path, optional): Path to save Ellie speech CSV.
        save_files (bool, optional): Whether to save the CSVs to disk (default True).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (participant_speech_df, ellie_speech_df)
    """
    
    all_participant_data = []
    all_ellie_data = []

    print(f"Starting transcript processing for {data_directory}...")

    for participant_id in label_df["Participant_ID"]:
        transcript_path = data_directory / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"

        if not transcript_path.exists():
            print(f"Transcript for participant {participant_id} not found. Skipping.")
            continue

        try:
            transcript_df = pd.read_csv(transcript_path, sep="\t")
            transcript_df["participant_id"] = participant_id

            participant_speech = transcript_df[transcript_df["speaker"] == "Participant"].copy()
            ellie_speech = transcript_df[transcript_df["speaker"] == "Ellie"].copy()

            all_participant_data.append(participant_speech)
            all_ellie_data.append(ellie_speech)

        except Exception as e:
            print(f"Error processing transcript for participant {participant_id}: {e}")

    # Combining results now
    final_participant_df = pd.concat(all_participant_data, ignore_index=True) if all_participant_data else pd.DataFrame()
    final_ellie_df = pd.concat(all_ellie_data, ignore_index=True) if all_ellie_data else pd.DataFrame()

    # Save if requested
    if save_files:
        if not final_participant_df.empty and participant_csv:
            final_participant_df.to_csv(participant_csv, index=False)
            print(f"Saved participant speech to: {participant_csv}")
        else:
            print("No participant data to save or path not provided.")

        if not final_ellie_df.empty and ellie_csv:
            final_ellie_df.to_csv(ellie_csv, index=False)
            print(f"Saved Ellie questions to: {ellie_csv}")
        else:
            print("No Ellie data to save or path not provided.")

    print("Processing complete.\n")
    return final_participant_df, final_ellie_df


    
def create_final_dataset(
    speech_csv_path: Path,
    labels_csv_path: Path,
    output_csv_path: Path,
    label_column: str
) -> pd.DataFrame:
    """
    Loads participant speech, aggregates it into single documents per participant,
    merges it with labels, and saves the final model-ready dataset.

    Args:
        speech_csv_path (Path): Path to the 'participant_speech.csv' file.
        labels_csv_path (Path): Path to the label file (e.g., train_split.csv).
        output_csv_path (Path): Path to save the final output CSV file.
        label_column (str): Name of the label column to use (e.g., 'phq8_binary' or 'phq_binary').

    Returns:
        pd.DataFrame: The final, merged, and cleaned DataFrame.
    """
    try:
        print(f"Reading speech data from: {speech_csv_path}")
        speech_df = pd.read_csv(speech_csv_path)
        
        print(f"Reading labels from: {labels_csv_path}")
        labels_df = pd.read_csv(labels_csv_path)

        # Normalizing label column name to 'label'
        if label_column != "label":
            labels_df = labels_df.rename(columns={label_column: "label"})

        print("Aggregating text for each participant...")
        aggregated_text_df = (
            speech_df.groupby("participant_id")["value"]
            .apply(lambda x: " ".join(x.astype(str)))
            .reset_index()
            .rename(columns={"value": "text"})
        )

        print("Merging aggregated text with labels...")
        merged_df = pd.merge(
            aggregated_text_df,
            labels_df,
            left_on="participant_id",
            right_on="Participant_ID",
            how="inner"
        )

        # Final dataset structure
        final_df = merged_df[["participant_id", "text", "label"]].copy()

        print(f"Saving final dataset to: {output_csv_path}")
        final_df.to_csv(output_csv_path, index=False)
        print("Successfully created final classification dataset.\n")

        return final_df

    except FileNotFoundError as e:
        print(f"Error: File not found. Details: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    
if __name__ == "__main__":
    print("Starting dataset preparation...\n")

    # Base paths
    BASE_PATH = Path("/Volumes/MACBACKUP/extracted_folders/")
    OUTPUT_PATH = Path("/Volumes/MACBACKUP/Regression")

    # Split files
    train_split_path = OUTPUT_PATH / "train_split_Depression_AVEC2017.csv"
    dev_split_path = OUTPUT_PATH / "dev_split_Depression_AVEC2017.csv"
    test_split_path = OUTPUT_PATH / "full_test_split.csv"

    # Load label files
    print("Loading label files...")
    train_df = pd.read_csv(train_split_path)
    dev_df = pd.read_csv(dev_split_path)
    test_df = pd.read_csv(test_split_path)

    # Process Transcripts (Train, Dev, Test)
    print("\nProcessing Transcripts")

    # TRAIN
    process_transcripts(
        data_directory=BASE_PATH,
        label_df=train_df,
        participant_csv=OUTPUT_PATH / "train_participant_speech.csv",
        ellie_csv=OUTPUT_PATH / "train_ellie_questions.csv",
        save_files=True
    )

    # DEV
    process_transcripts(
        data_directory=BASE_PATH,
        label_df=dev_df,
        participant_csv=OUTPUT_PATH / "dev_participant_speech.csv",
        ellie_csv=OUTPUT_PATH / "dev_ellie_questions.csv",
        save_files=True
    )

    # TEST
    process_transcripts(
        data_directory=BASE_PATH,
        label_df=test_df,
        participant_csv=OUTPUT_PATH / "test_participant_speech.csv",
        ellie_csv=OUTPUT_PATH / "test_ellie_questions.csv",
        save_files=True
    )
    
    # Create Final Datasets (Train, Dev, Test)
    print("\nCreating Final Datasets")

    # TRAIN
    create_final_dataset(
        speech_csv_path=OUTPUT_PATH / "train_participant_speech.csv",
        labels_csv_path=train_split_path,
        output_csv_path=OUTPUT_PATH / "final_datasets/final_train_dataset.csv",
        label_column="PHQ8_Score"
    )

    # DEV
    create_final_dataset(
        speech_csv_path=OUTPUT_PATH / "dev_participant_speech.csv",
        labels_csv_path=dev_split_path,
        output_csv_path=OUTPUT_PATH / "final_datasets/final_dev_dataset.csv",
        label_column="PHQ8_Score"
    )

    # TEST
    create_final_dataset(
        speech_csv_path=OUTPUT_PATH / "test_participant_speech.csv",
        labels_csv_path=test_split_path,
        output_csv_path=OUTPUT_PATH / "final_datasets/final_test_dataset.csv",
        label_column="PHQ_Score"
    )

    print("\nAll datasets processed and saved successfully!")