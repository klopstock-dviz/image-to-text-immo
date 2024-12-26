import ast
import pandas as pd
import ollama
import ast
import requests
from io import BytesIO
from time import time as timing
import traceback
import subprocess
import os

def main():
    username="klopstock-dviz"
    url = f'https://raw.githubusercontent.com/{username}/image-to-text-immo/main/_df_image_to_text.csv'

    df_image_to_text=pd.read_csv(url)
    display(df_image_to_text.head())
    display(df_image_to_text.shape)



    # save df on github
    def git_push(filename):
        token = os.getenv('GITHUB_TOKEN')
        command = f"""
        cd /image-to-text-immo && \
        git add {filename}.csv && \
        git diff-index --quiet HEAD || git commit -m 'Add processed DataFrame' && \
        git push https://{username}:{token}@github.com/{username}/image-to-text-immo.git
        """
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            print("No changes to commit or push failed.")
        else:
            print("Changes committed and pushed successfully.")




    web_root='https://raw.githubusercontent.com/klopstock-dviz/image-to-text-immo/refs/heads/main/photos/'
    image_to_text=[]

    model='llama3.2-vision:11b'

    # boucle sur df
    annonnces_traitees=['apimo-83969433', 'ag755493-435776219',
        'netty-cimmimmo-house-2339', 'adapt-immo-85002110433',
        'ag064863-440267846', 'hektor-1488_EXPERTIMO22-151783',
        'century-21-202_470_10547', 'keller-williams-1-34_1_8-165168',
        'keller-williams-1-34_1_8-165182', 'hektor-agim-mauriac-1988',
        'laforet-immo-facile-22411509',
        'citya-immobilier-5002-GES90220007-78', 'safti-1-1315900',
        'ag772915-442698362', 'square-habitat-immo-facile-42942232',
        'immo-facile-56901383', 'safti-1-1370112', 'immo-facile-33572968',
        'century-21-202_3122_4720', 'immo-facile-56307518',
        'citya-immobilier-10245-GES00340101-79', 'immo-facile-55922847',
        'iad-france-1621587', 'hektor-br-immobilier-4168',
        'immo-facile-57105018',]
    step_process_photo=0
    step_process_ad=0
    for i in range(0, len(df_image_to_text)):
    #for i in range(0, 3):
        print(f"annonce {step_process_ad}")
        annonce=df_image_to_text.loc[i]
        idannonce=annonce["idannonce"]

        if idannonce not in annonnces_traitees:

            # boucle sur photos
            paths_photos=ast.literal_eval(annonce["path_photos"])
            
            #for path in paths_photos[:3]:
            for path in paths_photos:
                try:
                    url_img= f"{web_root}{path.replace('/mnt/My Book2/Data/Immo/photos/','')}"
                    # print(url_img)
                    photo=path[path.find("photo_"):].replace(".jpg","")

                    response = requests.get(url_img)
                    # print(f"status request: {response.status_code}")
                    image_data = BytesIO(response.content).getvalue() 
                    
                    # Build the messages array
                    messages = [
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that describes real estate photos."
                        },
                        {
                            "role": "user", 
                            # "content": "Describe briefly this real estate photo, including its layout, features, and notable objects. Provide specific information about the materials, colors, and textures used throughout the space. Highlight any unique or distinctive elements that make this property stand out.",
                            "content":"Describe the scene depicted in this real estate photo.",
                            "images": [image_data]}
                    ]
                    
                    
                    t=timing()
                    stream = ollama.chat(
                        model=model,
                        messages=messages,
                        stream=True,
                        options={"temperature": 0.1}
                    )
                    
                    
                    response = ""
                    for chunk in stream:
                        content = chunk['message']['content']
                        response += content
                        # print(content, end='', flush=True)

                    time_photo_process=timing()-t
                    # print(f"exec time {time_photo_process}")
                    
                    image_to_text.append({
                        'idannonce': idannonce,
                        "photo": photo,
                        "url_img": url_img,
                        "content": response,
                        "model": model,
                        'process': 'ok',
                        "time_photo_process": time_photo_process
                    })

                    # save to drive each n photos processed
                    if (step_process_photo + 1) % 50 == 0:
                        filename="image_to_text_final_llama3-11b-vision"
                        csv_path = f'/image-to-text-immo/{filename}.csv'
                        pd.DataFrame(image_to_text).to_csv(csv_path, index=False)
                        git_push("image_to_text_final_llama3-11b-vision")
                        print(f"save to {csv_path}")

                    
                    step_process_photo+=1

                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    
                    image_to_text.append({
                        'idannonce': idannonce,
                        "photo": photo,
                        "url_img": url_img,
                        "content": str(e),
                        "model": model,
                        'process': 'ko',
                        "time_photo_process": 0
                    })
            step_process_ad+=1


    pd.DataFrame(image_to_text).to_csv(csv_path, index=False)
    git_push("image_to_text_final_llama3-11b-vision")

main()