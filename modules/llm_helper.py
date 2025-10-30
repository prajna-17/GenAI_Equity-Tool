from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
from typing import Optional, List, Dict
from huggingface_hub.errors import BadRequestError
import json
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np
import pandas as pd
from modules.fetch_news import get_financial_news # Assuming this file exists
from modules.fetch_stock import get_stock_history # Assuming this file exists
from config.settings import NEWS_API_KEY, HF_TOKEN



# choose a chat-capable model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 

# Global model for embeddings (loaded once)
try:
    # Use a small, fast model for local embeddings
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Could not load Sentence Transformer model: {e}")
    EMBEDDING_MODEL = None


# resilient hf_chat wrapper
def hf_chat(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """
    Try chat_completion first. If the model is NOT a chat model (BadRequest),
    fallback to text_generation. Return the final text string.
    """
    client = InferenceClient(MODEL_NAME, token=HF_TOKEN)

    # Try chat interface first (for chat-capable models)
    try:
        messages = [
            {"role": "system", "content": "You are a concise, factual financial assistant."},
            {"role": "user", "content": prompt}
        ]
        # Decrease temperature for RAG and JSON (structured) prompts
        current_temperature = temperature if "JSON format exactly" not in prompt and "Use ONLY the provided context" not in prompt else 0.2
        
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=current_temperature)
        return resp.choices[0].message["content"]

    except BadRequestError as e:
        # If model doesn't support chat, fallback to text generation
        try:
            tg = client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature)
            if isinstance(tg, str):
                return tg
            if hasattr(tg, "generated_text"):
                return tg.generated_text
            if isinstance(tg, dict):
                if "generated_text" in tg:
                    return tg["generated_text"]
                if "outputs" in tg and isinstance(tg["outputs"], list) and "generated_text" in tg["outputs"][0]:
                    return tg["outputs"][0]["generated_text"]
            return json.dumps(tg)
        except Exception as e2:
            raise RuntimeError(f"Both chat_completion and text_generation failed: chat_err={e}, textgen_err={e2}")

    except Exception as e_other:
        raise


# ------------ Prompt templates & simple chains ---------------

def summarize_news(article_text: str) -> str:
    """Return a short 3-sentence summary focused on facts and impact."""
    prompt = (
        "Summarize the following financial news in 3 short sentences focusing on: "
        "what happened, the company mentioned, and likely market impact.\n\n"
        f"Article:\n{article_text}"
    )
    return hf_chat(prompt, max_tokens=250)

def sentiment_of_news(article_text: str) -> str:
    """Return one-word sentiment and a short rationale: Bullish / Bearish / Neutral."""
    prompt = (
        "Analyze the news below and output the result in a single, valid JSON object exactly:\n"
        '{"sentiment":"<Bullish|Bearish|Neutral>", "reason":"A brief, direct explanation of market impact."}\n'
        f"News:\n{article_text}"
    )
    return hf_chat(prompt, max_tokens=150, temperature=0.2)


def summarize_live_news(company):
    """
    Fetch and summarize live financial news for a given company.
    """
    news_list = get_financial_news(company)
    summaries = []

    for n in news_list:
        content = n["description"]
        if not content:
            continue

        summary = summarize_news(content)

        # Call sentiment_of_news and try to parse the JSON
        sentiment_result = sentiment_of_news(content)
        try:
            parsed_sentiment = json.loads(sentiment_result)
            sentiment_text = parsed_sentiment.get("sentiment", "Neutral")
        except:
            # Fallback if JSON parsing fails
            sentiment_text = sentiment_result
        
        summaries.append({
            "title": n["title"],
            "content": content, 
            "summary": summary.strip(),
            "sentiment": sentiment_text.strip(),
            "source": n["source"],
            "url": n["url"]
        })
    
    return summaries


def merge_sentiment_and_price(sentiments: list, ticker: str):
    """
    Merge summarized sentiment results with recent stock price data.
    Returns a cleaned DataFrame with date, close price, sentiment, and sentiment_score.
    """

    # --- 1️⃣ Fetch stock price data ---
    try:
        price_df = get_stock_history(ticker, period="7d", interval="1d")
    except Exception as e:
        print(f"[Error] Could not fetch stock data for {ticker}: {e}")
        return pd.DataFrame()

    if price_df.empty or "date" not in price_df.columns:
        print(f"[Warning] No valid stock data found for {ticker}.")
        return pd.DataFrame()

    # Standardize Close column
    if "Close" not in price_df.columns:
        close_col = [c for c in price_df.columns if c.lower() == "close"]
        if close_col:
            price_df["Close"] = price_df[close_col[0]]
        else:
            price_df["Close"] = None

    # --- 2️⃣ Clean and normalize sentiments ---
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    cleaned_sentiments = []

    for item in sentiments:
        s = str(item.get("sentiment", "Neutral")).strip().lower()
        if "bull" in s:
            s = "Bullish"
        elif "bear" in s:
            s = "Bearish"
        else:
            s = "Neutral"
        cleaned_sentiments.append(s)

    numeric_scores = [score_map.get(s, 0) for s in cleaned_sentiments]
    avg_sentiment_score = sum(numeric_scores) / max(len(numeric_scores), 1)
    dominant_sentiment = (
        max(set(cleaned_sentiments), key=cleaned_sentiments.count)
        if cleaned_sentiments
        else "Neutral"
    )

    # --- 3️⃣ Create sentiment DataFrame ---
    today = pd.Timestamp.now().normalize()
    sentiment_df = pd.DataFrame({
        "date": [today],
        "sentiment": [dominant_sentiment],
        "sentiment_score": [avg_sentiment_score]
    })

    # --- 4️⃣ Merge with price data ---
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.tz_localize(None)

    merged = pd.merge(price_df, sentiment_df, on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0)
    merged["sentiment"] = merged["sentiment"].fillna("Neutral")

    # --- 5️⃣ Return merged DataFrame for visualization ---
    return merged

# ----------------- RAG Functions -----------------

def create_faiss_index_from_docs(documents: List[str]) -> Optional[faiss.Index]:
    """Helper function to create FAISS index from already filtered document texts."""
    if not EMBEDDING_MODEL or not documents:
        return None
    try:
        embeddings = EMBEDDING_MODEL.encode(documents)
        vector_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(vector_dim)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None


def run_rag_query(news_list: List[Dict], query: str, ticker: str, sources_to_include: Optional[List[str]] = None) -> str:
    """
    Performs RAG with dynamic source filtering: searches the indexed news summaries 
    and uses the context from only the specified sources to answer the query.
    """
    
    # 1. Apply filtering based on sources_to_include parameter
    filtered_news = news_list
    if sources_to_include and len(sources_to_include) < len(set(n.get('source') for n in news_list)):
        filtered_news = [
            n for n in news_list 
            if n.get('source') in sources_to_include
        ]

    # Prepare document texts from the filtered list (using summary and content)
    document_texts = [
        (n.get('summary', '') + " " + n.get('content', '')).strip()
        for n in filtered_news if n.get('summary') or n.get('content')
    ]

    # 2. Create the index from the filtered documents
    index = create_faiss_index_from_docs(document_texts)

    if not index or not document_texts:
        if sources_to_include and not filtered_news:
            return f"No news available from the selected sources ({', '.join(sources_to_include)}) to answer your question."
        return f"Could not process news for {ticker} to answer the question. No valid news summaries found."

    try:
        # 3. Embed the query
        query_embedding = EMBEDDING_MODEL.encode([query])
        
        # 4. Search the index (k=8 means retrieve the top 8 most relevant documents)
        k = min(8, len(document_texts)) # Max K is the number of documents available
        D, I = index.search(np.array(query_embedding).astype('float32'), k=k)
        
        # 5. Compile the context from the retrieved documents
        retrieved_context = "\n---\n".join([document_texts[i] for i in I[0] if i < len(document_texts)])

        # 6. Build the RAG Prompt
        prompt = f"""
        You are a financial analyst. Use ONLY the provided context to answer the user's question about {ticker}.
        Be factual, concise, and base your answer primarily on the provided news context. 
        If the context gives hints or related information, use reasonable inference. 
        Only say you cannot find the information if there is truly nothing relevant.


        USER QUESTION: {query}

        --- CONTEXT FROM NEWS SUMMARIES ---
        {retrieved_context}
        """

        # 7. Call the LLM with the context and low temperature for factual output
        return hf_chat(prompt, max_tokens=350, temperature=0.1)

    except Exception as e:
        return f"An error occurred during RAG query: {e}"
