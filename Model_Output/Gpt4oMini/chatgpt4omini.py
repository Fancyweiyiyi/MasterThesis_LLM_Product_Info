import openai
import yaml
import json
import re

# Load the configuration file
with open("config.yaml") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

openai.api_key = config_yaml['token']

# Load dataset as a generator (for memory efficiency)
def load_dataset(file_path):
    with open(file_path, "r") as f:
        for line in f:
            line = line.replace('NaN', 'null')  # Replace NaN with null
            yield json.loads(line)

# Clean JSON response
def clean_json_string(json_string):
    json_string = re.sub(r'[^\x20-\x7E]', '', json_string)  # Remove non-printable chars
    return json_string.strip()

dataset = load_dataset("NewDataset/office_products.jsonl")
results = []

# Process each entry in the dataset
for entry in dataset:
    title = entry.get('title', '')
    description = entry.get('description', '')

    if not title or not description:
        print("Missing title or description, skipping entry.")
        continue

    messages = [
        {
            "role": "system",
            "content": (
                "You are a world-class algorithm for extracting informtaion in strucured formats."
                "You support e-commerce customers by extracting attribute-value paris from product offers."
                "These attribute-value paris should be useful to compare product offers with the same product category. Large paragraphs of \"description\" are not attributes."
                "You should respond with a JSON object."
            )
        },
        {
            "role": "user",
            "content": f'Product title: "{title}"\nProduct description: "{description}"'
        }
    ]

    try:
        ans = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        continue

    raw_response = ans['choices'][0]['message']['content']
    print(f"Raw Response: {raw_response}")

    # Clean and decode the raw response
    try:
        cleaned_response = clean_json_string(raw_response)
        model_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON after cleaning: {e}")
        print(f"Raw Response (cleaned): {cleaned_response}")
        model_response = {}

    # Store results
    results.append({
        "title": title,
        "description": description,
        "model_response": model_response
    })

    # Print the results
    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Model Response: {model_response}")
    print('--------------------------------------------------\n')

# Save the results to a JSON file
with open("results1_off2_chatgpt4omini.json", "w") as f:
    json.dump(results, f, indent=4)
