from openai import OpenAI
import base64
import os
import json

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

episode_dir = "examples/images/select_1458/1pXnuDYAj8r/A1BZNPQ0H7ZSER:3GA6AFUKOQRGDQBWA7LPMGIOSJ0H31_sofa"
target_object = "sofa"


# Initialize an empty list to hold the data
data_template = []

# Function to add a new entry to the data template
def add_entry(entry_id, human_question, gpt_answer):
    entry = {
        "id": entry_id,
        "answer": [
            {
                "from": "human",
                "value": human_question
            },
            {
                "from": "gpt-4o-mini",
                "value": gpt_answer
            }
        ]
    }
    data_template.append(entry)

list_of_images = os.listdir(episode_dir+"/images/rgb")
list_of_images.sort()

for i in range(0, len(list_of_images) - 4, 3):  # Stops 4 images before the end to have a complete group of 5
    current_group = list_of_images[i:i + 5]

    # encode each
    base64_images = [encode_image(episode_dir+"/images/rgb/"+image) for image in current_group]

    human_question = f"Summarize the movable objects observed during the navigation as shown in the images. Please limit your response to one or two sentences focusing only on movable items."
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": human_question
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[1]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[2]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[3]}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[4]}"
                }
                }
            ]
            }
        ],
        max_tokens=300
        )

    print(response.choices[0].message.content)
    # 0-4, 3-7, 6-10, 9-13, 12-16
    add_entry(current_group[-1].split(".")[0],
              human_question,
            response.choices[0].message.content)

    # delete this json if already exist
    if os.path.exists(episode_dir+"/qa_history.json"):
        os.remove(episode_dir+"/qa_history.json")
    with open(episode_dir+"/qa_history.json", "w") as f:
        json.dump(data_template, f, indent=4)

    pass

