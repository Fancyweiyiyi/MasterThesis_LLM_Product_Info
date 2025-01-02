import openai
import pandas as pd
import json
import os

# Set your GLHF API key directly or via environment variable
openai.api_key = "glhf_bdc5347abfdebcb8a7f1e57e47810bb1"  # Directly set API key
openai.api_base = "https://glhf.chat/api/openai/v1"

# File paths for the uploaded datasets
csv_files = {
    "grocery_and_gourmet": "NewDataset/grocery_and_gourmet.csv",
    "home_and_garden": "NewDataset/home_and_garden.csv",
    "jewelry": "NewDataset/jewelry.csv",
    "office_products": "NewDataset/office_products.csv"
}

# Output directory for results
output_dir = "llama3_api_results"
os.makedirs(output_dir, exist_ok=True)

# Function to process a single file and save results
def process_file(file_path, category, output_dir):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Check if required columns exist
        if 'title' not in data.columns or 'description' not in data.columns:
            print(f"Missing 'title' or 'description' column in {file_path}. Skipping.")
            return

        # Initialize results list
        results = []
        
        # Iterate over rows in the dataset
        for index, row in data.iterrows():
            title = row['title']
            description = row['description']
            
            # Create the API request
            try:
                completion = openai.ChatCompletion.create(
                    stream=True,
                    model="hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                    messages=[
                        {"role": "system", "content": (
                            "You are a world-class algorithm for extracting information in structured formats. "
                            "You support e-commerce customers by extracting attribute-value pairs from product offers. "
                            "These attribute-value pairs should be useful to compare product offers with the same product category. Large paragraphs of \"description\" are not attributes. "
                            "You should respond with a JSON object."
                        )},
                        {"role": "user",
                         "content": f'Product title: "{title}"\nProduct description: "{description}"'}
                    ],
                )
                
                # Collect streamed responses
                response_text = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_text += chunk.choices[0].delta.content

                results.append({"title": title, "description": description, "response": response_text})
            
            except Exception as e:
                print(f"Error processing row {index} in {file_path}: {e}")
                results.append({"title": title, "description": description, "error": str(e)})
        
        # Save results to a JSON file
        output_file = os.path.join(output_dir, f"{category}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results for {category} saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Process each file
for category, file_path in csv_files.items():
    process_file(file_path, category, output_dir)