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

def save_data(df, filename, mode="w", log=True):
    """Save DataFrame to a CSV file."""
    try:
        # csv_path = f'/image-to-text-immo/{filename}.csv'
        csv_path = f'./{filename}.csv'
        df.to_csv(csv_path, index=False, mode=mode)
        if log:
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

def main():
    start_ollama_service()
    df_description_automatique_annonces = load_data(description_automatique_annonces_URL)

    logging.info(df_description_automatique_annonces.head())
    logging.info(df_description_automatique_annonces.shape)

    df_description_automatique_annonces.dropna(subset=["idannonce"], axis=0, inplace=True)


    logging.info(f"nb annonces à traiter: {len(df_description_automatique_annonces)}")
    df_description_automatique_annonces['resume_fr']=""
    df_description_automatique_annonces["time"]=0

    annonces_ids=['immo-facile-55859279', 'keller-williams-1-34_1_10-163580',
       'ag311656-417952452', 'immo-facile-40122976', 'iad-france-1684763',
       'ladresse-1-adresse-13764863', 'iad-france-1621850',
       'nestenn-1-99938876096', 'adapt-immo-0601812166',
       'immo-facile-55855885', 'safti-1-1353442', 'iad-france-1647593',
       'immo-facile-56980345', 'keller-williams-1-34_1_8-169653',
       'century-21-202_3574_679', 'hektor-chaigneau77-439',
       'immo-facile-55088933', 'ag311656-416851269', 'era-2-55525383',
       'apimo-85151896', 'century-21-202_3235_3180',
       'keller-williams-1-34_1_32-178172',
       'citya-immobilier-41039H-GES21840002-269', 'immo-facile-49167905',
       'immo-facile-53745236', 'century-21-202_6_23070',
       'ag922167-445804394', 'gedeon-30114825', 'ag440414-424172716',
       'ag880308-439062839', 'century-21-202_442_3938',
       'nestenn-1-99938816639', 'apimo-85067944',
       'keller-williams-1-34_1_16-174820',
       'citya-immobilier-1030-GES51660014-229', 'immo-facile-55404434',
       'safti-1-1353095', 'century-21-202_2986_7531',
       'citya-immobilier-5054-GES51470090-208',
       'citya-immobilier-34745045400017-GES57440019-597',
       'citya-immobilier-5002-TAPP529548', 'century-21-202_3273_1376',
       'apimo-83020460', 'hektor-macylienimmobilier-921',
       'century-21-202_6_23125', 'ag340369-442124852',
       'hektor-1809_EXPERTIMO22-174016', 'ag753199-403285848',
       'netty-sitbon-appt-49831', 'ag752345-445427069',
       'hektor-creteilvajou-1752', 'immo-facile-56395667',
       'apimo-85305634', 'iad-france-1684169', 'gedeon-30115469',
       'ag064863-434270240', 'immo-facile-54501429',
       'hektor-SIAMO94300-649', 'era-2-56459730', 'iad-france-1682038',
       'guy-hoquet-immo-facile-6377820', 'ag921172-442991960',
       'ag340369-443678636', 'apimo-85319582',
       'laforet-immo-facile-21471008', 'hektor-porchefontaine-104',
       'century-21-202_3655_235', 'apimo-84609436', 'ag311656-442066550',
       'ag783267-430576403', 'hektor-paris_est-490',
       'century-21-202_2643_5499', 'hektor-accord-187',
       'immo-facile-55282320', 'netty-comimob-appt-10132'
    ]

    step_process_ad=0
    for idx in df_description_automatique_annonces.index:
        try:
            t=timing()
            idannonce= df_description_automatique_annonces.loc[idx, 'idannonce']
            
            if idannonce not in annonces_ids:
                resume=df_description_automatique_annonces.loc[idx, "resume"]
                stream =  ollama.generate(
                    model=MODEL,
                    prompt=f"""Translate the following summary to French in no more than 1000 words. 
                        Do not include any additional commentary or repetition:
                        {resume}
                        End your response with the phrase: 'Translation complete.'""",
                    stream=True,
                    options={"temperature": 0.2, "max_tokens": 1400}
                )

                response = ""
                # init file
                save_data(pd.DataFrame([{"token_id": 0, "response": response}]), CURRENT_RESPONSE, 'w')

                token_id=0
                MAX_LENGTH = 1400 
                for chunk in stream:
                    if 'response' in chunk:
                        content = chunk['response']
                        response += content
                        token_id+=1

                    if (token_id + 1) % 300 == 0:
                        logging.info(f"log at token id {token_id}")
                        save_data(pd.DataFrame([{"id": token_id, "token": content, "time": datetime.datetime.now()}]), TOKEN_STATUS, "w", False)
                        save_data(pd.DataFrame([{"token_id": token_id, "response": response, "time": datetime.datetime.now()}]), CURRENT_RESPONSE, 'a', False)
                    # print(content, end='', flush=True)

                        # stop mechanism
                    if "Translation complete." in response:
                        response = response.split("Translation complete.")[0].strip()
                        break 
                                            
                    if token_id > MAX_LENGTH:
                        logging.warning("Token limit exceeded. Truncating response.")
                        response+=response+" | réponse tronquée"
                        break
                    
                df_description_automatique_annonces.loc[idx, "resume_fr"]=response
                df_description_automatique_annonces.loc[idx, "time"]=timing()-t
                    

                # save each n steps
                if (step_process_ad + 1) % 10 == 0:
                    _df_description_automatique_annonces=df_description_automatique_annonces[df_description_automatique_annonces["resume_fr"]!=""]
                    csv_path = save_data(_df_description_automatique_annonces[["idannonce", "resume_fr", 'time']], OUTPUT_FILENAME)
                    git_push(OUTPUT_FILENAME)
                
                logging.info(f"step {step_process_ad}----------------\n {idannonce}: {len(response.split(' '))} mots")
                step_process_ad += 1

                
                time.sleep(2)  # 1-second delay between API calls


        except Exception as e:
            logging.error(f"Exception occurred at index {idx}: {str(e)}", exc_info=True)
            log_exception()

        finally:
            logging.info("Streaming generator closed.")
    _df_description_automatique_annonces=df_description_automatique_annonces[df_description_automatique_annonces["resume_fr"]!=""][["idannonce", "resume_fr", 'time']]
    csv_path = save_data(_df_description_automatique_annonces, OUTPUT_FILENAME)
    git_push(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
