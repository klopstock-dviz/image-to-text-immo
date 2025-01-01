import pandas as pd
import ollama
from time import time as timing
import time
import traceback
import subprocess
import os
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='output.log', filemode='a')

# Configuration
USERNAME = "klopstock-dviz"
REPO_URL = f'https://github.com/{USERNAME}/image-to-text-immo.git'
description_automatique_annonces_URL = f'https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/description_automatique_annonces_en.csv'
MODEL = 'qwen2.5:7b'
OUTPUT_FILENAME = "description_automatique_annonces_translated"
TOKEN_STATUS="token_status"
CURRENT_RESPONSE="current_response"


def log_exception():
    # Capture the exception traceback as a string
    tb_str = traceback.format_exc()
    # Log the traceback string as an error
    logging.error(tb_str)


def load_data(url):
    """Load data from a URL into a DataFrame."""
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def save_data(df, filename, mode="w"):
    """Save DataFrame to a CSV file."""
    try:
        # csv_path = f'/image-to-text-immo/{filename}.csv'
        csv_path = f'./{filename}.csv'
        df.to_csv(csv_path, index=False, mode=mode)
        logging.info(f"Data saved to {csv_path}")
        return csv_path
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def git_push(filename):
    """Push changes to GitHub."""
    token = os.getenv('GITHUB_TOKEN')
    command = f"""
    ## cd /image-to-text-immo && \
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

def main():
    start_ollama_service()
    df_description_automatique_annonces = load_data(description_automatique_annonces_URL)

    logging.info(df_description_automatique_annonces.head())
    logging.info(df_description_automatique_annonces.shape)

    df_description_automatique_annonces.dropna(subset=["idannonce"], axis=0, inplace=True)


    logging.info(f"nb annonces à traiter: {len(df_description_automatique_annonces)}")
    df_description_automatique_annonces['resume_fr']=""

    step_process_ad=0
    for idx in df_description_automatique_annonces.index:
        try:
            idannonce= df_description_automatique_annonces.loc[idx, 'idannonce']
            resume=df_description_automatique_annonces.loc[idx, "resume"]
            stream =  ollama.generate(
                model=MODEL,
                prompt=f"""Translate this summary to French: {resume}\n\n
                        Stop when you finish the translation.""",
                stream=True,
                options={"temperature": 0.2, "stop": ["---", "Stop when you finish"], "max_tokens": 2000}
            )

            response = ""
            # init file
            save_data(pd.DataFrame([{"token_id": 0, "response": response}]), CURRENT_RESPONSE, 'w')

            token_id=0
            for chunk in stream:
                if 'response' in chunk:
                    content = chunk['response']
                    response += content
                    token_id+=1

                if (token_id + 1) % 200 == 0:
                    logging.info(f"log at token id {token_id}")
                    save_data(pd.DataFrame([{"id": token_id, "token": content, "time": datetime.datetime.now()}]), TOKEN_STATUS, "w")
                    save_data(pd.DataFrame([{"token_id": token_id, "response": response, "time": datetime.datetime.now()}]), CURRENT_RESPONSE, 'a')
                # print(content, end='', flush=True)

            df_description_automatique_annonces.loc[idx, "resume_fr"]=response
                

            # save each n steps
            if (step_process_ad + 1) % 5 == 0:
                _df_description_automatique_annonces=df_description_automatique_annonces[df_description_automatique_annonces["resume_fr"]!=""]
                csv_path = save_data(_df_description_automatique_annonces[["idannonce", "resume_fr"]], OUTPUT_FILENAME)
                git_push(OUTPUT_FILENAME)
            
            logging.info(f"step {step_process_ad}----------------\n {idannonce}: {len(response.split(' '))} mots")
            step_process_ad += 1

            
            time.sleep(2)  # 1-second delay between API calls


        except Exception as e:
            logging.error(f"Exception occurred at index {idx}: {str(e)}", exc_info=True)
            log_exception()

        finally:
            logging.info("Streaming generator closed.")
    _df_description_automatique_annonces=df_description_automatique_annonces[df_description_automatique_annonces["resume_fr"]!=""][["idannonce", "resume_fr"]]
    csv_path = save_data(_df_description_automatique_annonces, OUTPUT_FILENAME)
    git_push(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
