import ast
import pandas as pd
import ollama
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
df_image_to_text_URL = f'https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/_df_image_to_text.csv'
image_to_text_final_llama3_URL=f"https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/image_to_text_final_llama3-11b-vision.csv"
idannonces_summarized_URL=f"https://raw.githubusercontent.com/{USERNAME}/image-to-text-immo/main/idannonces_summarized.csv"
MODEL = 'qwen2.5:7b'
OUTPUT_FILENAME = "description_automatique_annonces"


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

def generate_prompt_template(structured_data):

    # Format structured data into JSON format
    structured_json = {
        "Property type": structured_data["typedebien"].values[0],
        "Transaction type": structured_data["typedetransaction"].values[0],
        "Municipality": {structured_data['ville'].values[0]},
        "Property price": f"{structured_data['prix_bien'].values[0]} EUR",        
        "Price per m2": f"{structured_data['prix_m_carre'].values[0]} EUR",
        "Surface area": f"{structured_data['surface'].values[0]} m²",
        "Number of pieces": structured_data["nb_pieces"].values[0]
    }

    # Define a template with combined information
    prompt_template = f"""
        You are a helpful assistant tasked with summarizing a property based on detailed descriptions of various rooms and spaces. Your task is to first extract the key features of each room or space, such as the overall setting, color schemes, architectural details, and any notable features. 
        Below is information about a property:
        Structured data: {structured_json}
        Description: look at the context history

        These are the precise taks you are to complete:
        1.Summarize each room / space desciption provided in the context history,
        
        2.Provide a final conclusion that includes the following:
        2.1.Bullet point summary for the entire housing following this format:
        - **Property type**: <Property type>
        - **Transaction type**: <Transaction type>
        - **Municipality**: <Municipality>
        - **Property price**: <Property price>
        - **Price per m2**: <Price per m2>
        - **Surface area**: <Surface area>
        - **Number of pieces**: <Pieces>
        - **Key points**: <Brief summary of key features, amenities and any unique points in the description>

        2.2.More detailled summary for the entire housing that emphazises on:
        The overall feel of the property.
        The design elements (e.g., minimalist, modern, classical, etc.).
        The state of the property (e.g., finished, unfinished, needs for renovation).        
        Ensure that the summary is neutral, brief, and informative, while retaining the essence of the property's current condition and potential.



        Output:

    """

    return prompt_template



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
    df_image_to_text = load_data(df_image_to_text_URL)
    image_to_text_final_llama3=load_data(image_to_text_final_llama3_URL)
    idannonces_summarized=load_data(idannonces_summarized_URL)

    logging.info(df_image_to_text.head())
    logging.info(df_image_to_text.shape)

    df_image_to_text.dropna(subset=["idannonce"], axis=0, inplace=True)
    image_to_text_final_llama3.dropna(subset=["idannonce"], axis=0, inplace=True)
    image_to_text_final_llama3=image_to_text_final_llama3[~image_to_text_final_llama3["idannonce"].isin(idannonces_summarized["idannonce"])]

    results=[]
    step_process_ad=0
    for idannonce in image_to_text_final_llama3["idannonce"].unique():
        t=timing()

        print("annonce", idannonce)
        df_arr_summaries=image_to_text_final_llama3[image_to_text_final_llama3["idannonce"]==idannonce]

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summaries descriptions of real estate various rooms."},
        ]    
        for i in df_arr_summaries.index:
            text=df_arr_summaries.loc[i, "content"]

            messages.append(
                {
                    "role": "user", 
                    "content": text
                }    
            )

        structured_data=df_image_to_text[df_image_to_text["idannonce"]==idannonce]
        messages.append({
            "role": "user",
            "content": generate_prompt_template(structured_data)
        })

        stream = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0.2}
        )

        response = ""
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            

        tf=timing()-t

        results.append(
            {
                "idannonce": idannonce,
                "messages": messages, 
                "resume": response, 
                "time": tf}
        )

        # save each n steps
        if (step_process_ad + 1) % 5 == 0:
            csv_path = save_data(pd.DataFrame(results), OUTPUT_FILENAME)
            git_push(OUTPUT_FILENAME)

        step_process_ad += 1
        logging.info(f"{idannonce}: {len(response.split(' '))} mots")

    csv_path = save_data(pd.DataFrame(results), OUTPUT_FILENAME)
    git_push(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
