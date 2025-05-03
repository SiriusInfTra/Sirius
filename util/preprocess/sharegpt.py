import pandas
import json

prompt_json = 'workload_data/sharegpt/sg_90k_part1_html_cleaned.json'
json_data = json.load(open(prompt_json))

prompts = [] 
outputs = []
conversation_idx = -1
in_count = 0
gen_count = 0
for data in json_data:
    prefix_conversation = ""
    for message in data['conversations']:
        if message["from"] == "human":
            conversation_idx += 1
            in_count += 1
            prefix_conversation += message["value"]
            prompts.append(prefix_conversation)
        elif message["from"] == "gpt":
            gen_count += 1
            prefix_conversation += message["value"]
            outputs.append(message["value"])
