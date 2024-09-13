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
# iterate over episode_dir
for image in list_of_images:
    image_id = image.split(".")[0]
    # Getting the base64 string
    base64_image = encode_image(episode_dir+"/images/rgb/"+image)

    human_question = f"Would a {target_object} be found here? Why or why not?"
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
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      max_tokens=300
    )

    add_entry(image_id, "Would a sofa be found here? Why or why not?",
              response.choices[0].message.content)

    # delete this json if already exist
    if os.path.exists(episode_dir+"/qa_CoT.json"):
        os.remove(episode_dir+"/qa_CoT.json")
    with open(episode_dir+"/qa_CoT.json", "w") as f:
        json.dump(data_template, f, indent=4)

    print(response.choices[0].message.content)

    pass
