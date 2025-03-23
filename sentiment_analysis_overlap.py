from transformers import pipeline
import numpy as np

# 1. Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Function to get sentiment scores
def get_sentiment_score(text):
    # Some sentiment analyzers return multiple results for longer texts
    # Let's limit the text length to avoid this issue
    text = text[:512]  # Most transformer models have a token limit
    
    result = sentiment_analyzer(text)
    
    # If only one result, extract the score
    if isinstance(result, dict):
        # Convert to positive sentiment score (0 to 1)
        return result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
    
    # If multiple results, average the sentiment scores
    positive_score = sum(r["score"] for r in result if r["label"] == "POSITIVE") / len(result)
    negative_score = sum(r["score"] for r in result if r["label"] == "NEGATIVE") / len(result)
    
    # Return positive sentiment proportion (0 to 1)
    if positive_score + negative_score > 0:
        return positive_score / (positive_score + negative_score)
    else:
        return 0.5  # Neutral

# 3. Function to calculate sentiment similarity
def sentiment_similarity(sentiment1, sentiment2):
    # Simple absolute difference, inverted so higher means more similar
    return 1.0 - abs(sentiment1 - sentiment2)

# 4. Apply functions to dataframe
# Extract sentiment scores
df['prompt_sentiment'] = df['prompt'].apply(get_sentiment_score)
df['response_a_sentiment'] = df['response_a'].apply(get_sentiment_score)
df['response_b_sentiment'] = df['response_b'].apply(get_sentiment_score)

# Calculate sentiment similarity
df['a_sentiment_match'] = df.apply(lambda row: sentiment_similarity(
    row['prompt_sentiment'], row['response_a_sentiment']), axis=1)
df['b_sentiment_match'] = df.apply(lambda row: sentiment_similarity(
    row['prompt_sentiment'], row['response_b_sentiment']), axis=1)

# 5. Get sentiment difference between models (which model has more similar sentiment)
df['sentiment_match_advantage_a'] = df['a_sentiment_match'] - df['b_sentiment_match']


################################## BATCH PROCESSING ##################################
# Batch processing for sentiment analysis
def batch_get_sentiment(texts, batch_size=32):
    all_sentiments = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_texts = [text[:512] for text in batch_texts]  # Truncate to avoid token limit issues
        
        results = sentiment_analyzer(batch_texts)
        
        # Process each result
        for result in results:
            sentiment_score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
            all_sentiments.append(sentiment_score)
    
    return all_sentiments

# Process all texts in batches
prompts = df['prompt'].tolist()
responses_a = df['response_a'].tolist()
responses_b = df['response_b'].tolist()

prompt_sentiments = batch_get_sentiment(prompts)
response_a_sentiments = batch_get_sentiment(responses_a)
response_b_sentiments = batch_get_sentiment(responses_b)

df['prompt_sentiment'] = prompt_sentiments
df['response_a_sentiment'] = response_a_sentiments
df['response_b_sentiment'] = response_b_sentiments

# Calculate sentiment similarities
df['a_sentiment_match'] = [1.0 - abs(p - r) for p, r in zip(prompt_sentiments, response_a_sentiments)]
df['b_sentiment_match'] = [1.0 - abs(p - r) for p, r in zip(prompt_sentiments, response_b_sentiments)]
df['sentiment_match_advantage_a'] = df['a_sentiment_match'] - df['b_sentiment_match']