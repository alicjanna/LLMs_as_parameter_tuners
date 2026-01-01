from anthropic import Anthropic
from openai import OpenAI
import openai
import google.generativeai as genai
from mistralai import Mistral
import pandas as pd
import json
import re

sheets = ['TSP-GA-GA', 'TSP-GA-SA', 'TSP-GA-ACO',
          'JSSP-GA', 'JSSP-PSO', 'JSSP-ACO',
          'GCP-PSO', 'GCP-GA', 'GCP-SA']

ant_api_key = ''
openai.api_key = ''
gemini_api_key = ''
mistral_api_key = ''


def get_claude_response(prompt, client):
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if message.content and len(message.content) > 0:
            return message.content[0].text
        else:
            return "No content found in the response."
    except Exception as e:
        return f"Error in API request: {str(e)}"

def get_chat_response(prompt):
    try:
        message = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if message.choices and len(message.choices) > 0:
            return message.choices[0].message.content
        else:
            return "No content found in the response."
    except Exception as e:
        return f"Error in API request: {str(e)}"


def get_gemini_response(prompt):
    try:
        model_ = genai.GenerativeModel('gemini-2.0-flash')
        response = model_.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in API request: {str(e)}"


def get_mistral_response(prompt, client):
    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        if chat_response.choices and len(chat_response.choices) > 0:
            return chat_response.choices[0].message.content
        else:
            return "No content found in the response."
    except Exception as e:
        return f"Error in API request: {str(e)}"


def get_llama_response(prompt):
    try:
        #  just a placeholder
        return True
    except Exception as e:
        return f"Error in API request: {str(e)}"


def get_response(row):
    model = row['llm']
    prompt = row['prompt']
    prompt += " Don't give me rationale, or any additional text. I want pure dictionary. And for each parameter give just ONE value."

    if model == 'Claude':
        client = Anthropic(api_key=ant_api_key)
        r = get_claude_response(prompt, client)
    elif model == 'GPT-4o':
        r = get_chat_response(prompt)
    elif model == 'Gemini':
        genai.configure(api_key=gemini_api_key)
        r = get_gemini_response(prompt)
    elif model == 'Mistral':
        client = Mistral(api_key=mistral_api_key)
        r = get_mistral_response(prompt, client)
    else:
        r = []
    return r


def give_me_dict(r):
    match = re.search(r'```python\n(.*?)```', r, re.DOTALL)

    if match:
        json_str = match.group(1)
        clean_str = json_str.replace('True', 'true').replace("False", "false")

        dict_match = re.search(r'param\s*=\s*({.*})', clean_str, re.DOTALL)
        if dict_match:
            json_str = dict_match.group(1)
            clean_str = json_str.replace('True', 'true').replace("False", "false")
            clean_str = re.sub(r',\s*}', '}', clean_str)
            clean_str = json.loads(clean_str)
            return clean_str
    else:
        clean_str = r.strip('`').lstrip('python\n')
        clean_str = clean_str.replace('True', 'true').replace("False", "false")
        clean_str = re.sub(r',\s*}', '}', clean_str)
        return json.loads(clean_str)



def extract_dict_from_string(r: str) -> dict:

    try:
        match = re.search(r'```(?:python)?\n(.*?)```', r, re.DOTALL)
        content = match.group(1) if match else r.strip()

        dict_match = re.search(r'({.*})', content, re.DOTALL)
        if not dict_match:
            raise ValueError("No dictionary structure found.")
        json_like = dict_match.group(1)

        json_like = json_like.replace("True", "true").replace("False", "false")
        json_like = re.sub(r",\s*}", "}", json_like)  # trailing commas
        json_like = re.sub(r",\s*\]", "]", json_like)  # trailing commas in lists
        json_like = json_like.replace("'", '"')

        return json.loads(json_like)

    except Exception as e:
        raise ValueError(f"Failed to parse dictionary: {e}")


### SCRIPT

responses = {}


for sheet in sheets:

    big_file = pd.read_excel('./parameters_main.xlsx', sheet_name=sheet)

    file = big_file.loc[(big_file['run'] == 'Feedback') & (big_file['llm'] != 'llama3')]

    for _, row in file.iterrows():
        r = get_response(row)
        configuration = extract_dict_from_string(r)
        responses[row['problem'] + '_' + row['MHA'] + '_' + row['instance'] + '_' + row['llm']] = configuration



processed = pd.DataFrame.from_dict(responses, orient='index')

processed.reset_index(inplace=True)
processed['problem'] = processed['index'].apply(lambda x: x.split('_')[0])
processed['MHA'] = processed['index'].apply(lambda x: x.split('_')[1])
processed['instance'] = processed['index'].apply(lambda x: x.split('_')[2])
processed['llm'] = processed['index'].apply(lambda x: x.split('_')[3])

processed.drop('index', axis=1, inplace=True)
processed.drop('mutation multipoints', axis=1, inplace=True)


#mer
df = pd.merge(big_file, processed, left_on=['problem', 'llm', 'instance', 'MHA'],
              right_on=['problem', 'llm', 'instance', 'MHA'], how='left')

df.to_csv('./params.csv')