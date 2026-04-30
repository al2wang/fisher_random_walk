#### Collect prompt data from CAIS/MMLU datasets

# !pip install datasets
from datasets import load_dataset
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
    

# cais/mmlu
subs = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics']
for sub in subs:
    dataset = load_dataset("cais/mmlu", sub)
    print(dataset)

    questions_df = pd.DataFrame(columns=['subject','question'])
    for i in range(100):
        example = dataset['test'][i]
        question = example['question']
        choices = example['choices']
        answer = example['answer']
        subject = example['subject']
        qt0 = f"{question}{choices}"
        questions_df = pd.concat([questions_df, pd.DataFrame([{'subject':sub, 'question': qt0}])], ignore_index=True)

    print(questions_df.shape)
    questions_df.to_csv(f'data/questions_{sub}.csv', index=False)

#### Decode the prompt data

# do embadding
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

client = OpenAI()

for sub in subs:
    # Read the CSV file
    csv_path = f'data/questions_{sub}.csv'
    df = pd.read_csv(csv_path)
    
    # Ensure the 'combined' column exists or adjust as needed
    if 'question' not in df.columns:
        raise ValueError(f"The 'combined' column does not exist in the DataFrame for {sub}")
    
    # Apply the embedding function to the 'combined' column
    tqdm.pandas()  # Add a progress bar to the lambda function
    df['embedding'] = df['question'].progress_apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    
    # Save the DataFrame with embeddings to a new CSV file
    df.to_csv(f'data/questions_{sub}.csv', index=False)

print("Embedding completed and saved for all subjects.")