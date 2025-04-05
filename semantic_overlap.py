import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


df = pd.read_csv("train.csv")

def get_embedding(text, tokenizer, model, device):
    # Tokenize input
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Get model output
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use the [CLS] token embedding as the sentence representation
    # or use mean pooling for a potentially better representation
    token_embeddings = model_output.last_hidden_state
    
    # Mean pooling - taking average of all token embeddings
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    return embedding[0]  # Return the first (and only) embedding

def semantic_overlap(prompt, response, tokenizer, model, device):
    # Get embeddings for prompt and response
    prompt_embedding = get_embedding(prompt, tokenizer, model, device)
    response_embedding = get_embedding(response, tokenizer, model, device)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([prompt_embedding], [response_embedding])[0][0]
    return similarity

# Load model and tokenizer once
model_name = "sentence-transformers/all-mpnet-base-v2"  # Good model for semantic similarity
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Apply to dataframe to create features
df['a_semantic_overlap'] = df.apply(
    lambda row: semantic_overlap(row['prompt'], row['response_a'], tokenizer, model, device), 
    axis=1
)

df['b_semantic_overlap'] = df.apply(
    lambda row: semantic_overlap(row['prompt'], row['response_b'], tokenizer, model, device), 
    axis=1
)

# save the dataframe with the new features
df.to_csv('df_with_semantic_overlap.csv', index=False)






# ######################################BATCH EMBEDDINGS#################################################################

# def get_batch_embeddings(texts, tokenizer, model, device, batch_size=32):
#     embeddings = []
    
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
#         encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
#         with torch.no_grad():
#             model_output = model(**encoded_input)
        
#         # Mean pooling
#         token_embeddings = model_output.last_hidden_state
#         attention_mask = encoded_input['attention_mask']
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
#         embeddings.append(batch_embeddings)
    
#     return np.vstack(embeddings)

# # Process all texts in batches
# prompts = df['prompt'].tolist()
# responses_a = df['response_a'].tolist()
# responses_b = df['response_b'].tolist()

# prompt_embeddings = get_batch_embeddings(prompts, tokenizer, model, device)
# response_a_embeddings = get_batch_embeddings(responses_a, tokenizer, model, device)
# response_b_embeddings = get_batch_embeddings(responses_b, tokenizer, model, device)

# # Calculate similarities
# similarities_a = [cosine_similarity([p], [r])[0][0] for p, r in zip(prompt_embeddings, response_a_embeddings)]
# similarities_b = [cosine_similarity([p], [r])[0][0] for p, r in zip(prompt_embeddings, response_b_embeddings)]

# df['a_semantic_overlap'] = similarities_a
# df['b_semantic_overlap'] = similarities_b