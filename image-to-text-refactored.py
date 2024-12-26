import ast
import pandas as pd
import ollama
import requests
from io import BytesIO
from time import time as timing
import traceback
import subprocess
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='output.log', filemode='a')

# Configuration
USERNAME = "klopstock-dviz"
REPO_URL = f'https://github.com/{USERNAME}/image-to-text-immo.git'
CSV_URL = f'https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/_df_image_to_text.csv'
WEB_ROOT = 'https://raw.githubusercontent.com/klopstock-dviz/image-to-text-immo/refs/heads/main/photos/'
MODEL = 'llama3.2-vision:11b'
OUTPUT_FILENAME = "image_to_text_final_llama3-11b-vision"

# List of processed announcements
ANNONCES_TRAITEES = [
    'apimo-83969433', 'ag755493-435776219', 'netty-cimmimmo-house-2339', 'adapt-immo-85002110433',
    'ag064863-440267846', 'hektor-1488_EXPERTIMO22-151783', 'century-21-202_470_10547',
    'keller-williams-1-34_1_8-165168', 'keller-williams-1-34_1_8-165182', 'hektor-agim-mauriac-1988',
    'laforet-immo-facile-22411509', 'citya-immobilier-5002-GES90220007-78', 'safti-1-1315900',
    'ag772915-442698362', 'square-habitat-immo-facile-42942232', 'immo-facile-56901383',
    'safti-1-1370112', 'immo-facile-33572968', 'century-21-202_3122_4720', 'immo-facile-56307518',
    'citya-immobilier-10245-GES00340101-79', 'immo-facile-55922847', 'iad-france-1621587',
    'hektor-br-immobilier-4168', 'immo-facile-57105018'
]

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

def process_image(url_img, model, photo, idannonce):
    """Process a single image and return the description."""
    try:
        response = requests.get(url_img)
        image_data = BytesIO(response.content).getvalue()

        messages = [
            {"role": "system", "content": "You are a helpful assistant that describes real estate photos."},
            {"role": "user", "content": "Describe the scene depicted in this real estate photo.", "images": [image_data]}
        ]

        t = timing()
        stream = ollama.chat(model=model, messages=messages, stream=True, options={"temperature": 0.1})

        response_text = ""
        for chunk in stream:
            response_text += chunk['message']['content']

        time_photo_process = timing() - t

        return {
            'idannonce': idannonce,
            "photo": photo,
            "url_img": url_img,
            "content": response_text,
            "model": model,
            'process': 'ok',
            "time_photo_process": time_photo_process
        }
    except Exception as e:
        logging.error(f"Error processing image {url_img}: {e}")
        return {
            'idannonce': idannonce,
            "photo": photo,
            "url_img": url_img,
            "content": str(e),
            "model": model,
            'process': 'ko',
            "time_photo_process": 0
        }

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
    df_image_to_text = load_data(CSV_URL)
    logging.info(df_image_to_text.head())
    logging.info(df_image_to_text.shape)

    image_to_text = []
    step_process_photo = 0
    step_process_ad = 0

    for i in range(len(df_image_to_text)):
        annonce = df_image_to_text.loc[i]
        idannonce = annonce["idannonce"]

        if idannonce not in ANNONCES_TRAITEES:
            paths_photos = ast.literal_eval(annonce["path_photos"])

            for path in paths_photos:
                url_img = f"{WEB_ROOT}{path.replace('/mnt/My Book2/Data/Immo/photos/','')}"
                photo = path[path.find("photo_"):].replace(".jpg","")

                result = process_image(url_img, MODEL, photo, idannonce)
                image_to_text.append(result)

                if (step_process_photo + 1) % 50 == 0:
                    csv_path = save_data(pd.DataFrame(image_to_text), OUTPUT_FILENAME)
                    git_push(OUTPUT_FILENAME)

                step_process_photo += 1

            step_process_ad += 1

    csv_path = save_data(pd.DataFrame(image_to_text), OUTPUT_FILENAME)
    git_push(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
