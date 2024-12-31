import pandas as pd
import ollama
from time import time as timing
import time
import traceback
import subprocess
import os
import logging
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='output.log', filemode='a')

# Configuration
USERNAME = "klopstock-dviz"
REPO_URL = f'https://github.com/{USERNAME}/image-to-text-immo.git'
description_automatique_annonces_URL = f'https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/description_automatique_annonces_en.csv'
MODEL = 'qwen2.5:7b'
OUTPUT_FILENAME = "description_automatique_annonces_translated"
PROCESSED_IDS_FILE = "/image-to-text-immo/processed_ids.txt"
MAX_RETRIES = 3
TIMEOUT = 15  # Timeout for API calls (in seconds)

def log_exception():
    """Log the exception traceback."""
    tb_str = traceback.format_exc()
    logging.error(tb_str)

def load_processed_ids():
    """Load processed IDs from file."""
    if os.path.exists(PROCESSED_IDS_FILE):
        with open(PROCESSED_IDS_FILE, 'r') as file:
            return set(line.strip() for line in file.readlines())
    return set()

def save_processed_id(idannonce):
    """Append processed ID to file."""
    with open(PROCESSED_IDS_FILE, 'a') as file:
        file.write(f"{idannonce}\n")

def load_data(url):
    """Load data from a URL into a DataFrame."""
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def save_data(df, filename):
    """Save DataFrame to a CSV file."""
    try:
        csv_path = f'/image-to-text-immo/{filename}.csv'
        df.to_csv(csv_path, index=False)
        logging.info(f"Data saved to {csv_path}")
        return csv_path
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def git_push(filename):
    """Push changes to GitHub."""
    token = os.getenv('GITHUB_TOKEN')
    command = f"""
    cd /image-to-text-immo && \
    git add {filename}.csv && \
    git diff-index --quiet HEAD || git commit -m 'Add processed DataFrame' && \
    git push https://{USERNAME}:{token}@github.com/{USERNAME}/image-to-text-immo.git
    """
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        logging.error(f"Push failed: {result.stderr}")
    else:
        logging.info("Changes committed and pushed successfully.")

def start_ollama_service():
    """Start the ollama service if it is not running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True, text=True)
        if not result.stdout.strip():
            logging.info("Starting ollama service...")
            subprocess.Popen(['nohup', '/usr/local/bin/ollama', 'serve', '>', 'ollama.log', '2>&1', '&'])
        else:
            logging.info("ollama service is already running.")
    except Exception as e:
        logging.error(f"Error starting ollama service: {e}")

def generate_with_timeout(resume, model):
    """Generate a translation with a timeout and retry mechanism."""
    def generate():
        stream = ollama.generate(
            model=model,
            prompt=f"Translate this summary to french: {resume}",
            stream=True,
            options={"temperature": 0.2}
        )
        response = ""
        for chunk in stream:
            if 'response' in chunk:
                response += chunk['response']
        return response

    for attempt in range(MAX_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate)
                return future.result(timeout=TIMEOUT)
        except concurrent.futures.TimeoutError:
            logging.warning(f"Timeout occurred on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            logging.error(f"Error during generation on attempt {attempt + 1}: {str(e)}")
        time.sleep(2)  # Wait before retrying

    logging.error("Max retries exceeded. Skipping this entry.")
    return None

def main():
    start_ollama_service()
    processed_ids = load_processed_ids()
    df_description_automatique_annonces = load_data(description_automatique_annonces_URL)

    logging.info(df_description_automatique_annonces.head())
    logging.info(df_description_automatique_annonces.shape)

    df_description_automatique_annonces.dropna(subset=["idannonce"], axis=0, inplace=True)

    logging.info(f"nb annonces à traiter: {len(df_description_automatique_annonces)}")
    df_description_automatique_annonces['resume_fr'] = ""

    step_process_ad = 0
    for idx in df_description_automatique_annonces.index:
        try:
            idannonce = df_description_automatique_annonces.loc[idx, 'idannonce']

            # Skip already processed IDs
            if idannonce in processed_ids:
                logging.info(f"Skipping already processed idannonce: {idannonce}")
                continue

            resume = df_description_automatique_annonces.loc[idx, "resume"]
            response = generate_with_timeout(resume, MODEL)

            if response is None:
                logging.error(f"Skipping idx {idx} due to repeated failures.")
                continue

            df_description_automatique_annonces.loc[idx, "resume_fr"] = response

            # Save the processed ID
            save_processed_id(idannonce)

            # Save progress every 10 steps
            if (step_process_ad + 1) % 10 == 0:
                csv_path = save_data(df_description_automatique_annonces, OUTPUT_FILENAME)
                git_push(OUTPUT_FILENAME)

            logging.info(f"step {step_process_ad}----------------\n {idannonce}: {len(response.split(' '))} mots")
            step_process_ad += 1

            time.sleep(2)  # Delay between API calls

        except Exception as e:
            logging.error(f"Exception occurred at index {idx}: {str(e)}", exc_info=True)
            log_exception()

    csv_path = save_data(df_description_automatique_annonces, OUTPUT_FILENAME)
    git_push(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
