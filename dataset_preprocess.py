import os
import requests
import argparse
import pandas as pd

def download_and_save_video(url, storage_name, final_storage_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            filename = f"{storage_name}"
            full_filepath = os.path.join(final_storage_path, filename)
            with open(full_filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Downloaded and saved: {full_filepath} for URL: {url}")
        else:
            print(f"Failed to download video from {url}")
    except Exception as e:
        print(f"Failed to download video from {url}")

def process_csv_file(csv_file_path, storage_path):
    try: 
        df = pd.read_csv(csv_file_path, sep=';', skiprows=1, header=None, names=['source_name', 'url', 'final_storage_name', 'ts_multiplier'])
        for index, row in df.iterrows():
            try: 
                source_name = row['source_name'].strip()
                if not os.path.exists(f"{storage_path}/{source_name}/"):
                    os.makedirs(f"{storage_path}/{source_name}/")
                final_storage_path = f"{storage_path}/{source_name}/"
                url = row['url'].strip()
                final_storage_name = row['final_storage_name'].strip()
                ts_multiplier = float(row['ts_multiplier']) if pd.notna(row['ts_multiplier']) else 1.0
                print(f"Processing {url} with multiplier {ts_multiplier} and saving as {final_storage_name}")
                download_and_save_video(url, final_storage_name, final_storage_path)
            except Exception as e:
                print(f"Failed to process {url} with name as {final_storage_name} and multiplier {ts_multiplier}\n Exception occured: {e}")
    except Exception as e:
        print(f"Exception occured: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save videos from CSV")
    parser.add_argument("csv_file_path", help="CSV file containing source_name, url, final_storage_name, and optional ts_multiplier columns prefixed with crowd11/")
    parser.add_argument("final_storage_path", help="Directory where the videos will be saved prefixed with crowd11/")
    args = parser.parse_args()
    csv_file_path_parsed = "crowd11/" + str(args.csv_file_path)
    final_storage_path_parsed = "crowd11/" + str(args.final_storage_path)
    process_csv_file(csv_file_path_parsed, final_storage_path_parsed)