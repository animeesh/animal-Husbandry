import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

# Load Excel data
file_path = "your_excel_file.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Combine relevant columns into a single string for embedding
df['Combined'] = df['Name']  # Replace 'Name' with the column you want to process

# Generate embeddings using OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [e['embedding'] for e in response['data']]

# Generate embeddings for the combined column
df['Embeddings'] = get_embeddings(df['Combined'].tolist())

# Calculate similarity between all rows
def find_similar_rows(embeddings, threshold=0.8):
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > threshold:
                similarities.append((i, j, sim))
    return similarities

# Find similar rows
similarities = find_similar_rows(df['Embeddings'].tolist())

# Display similar rows
for pair in similarities:
    i, j, sim = pair
    print(f"Row {i} ('{df['Combined'][i]}') is similar to Row {j} ('{df['Combined'][j]}') with similarity: {sim:.2f}")
