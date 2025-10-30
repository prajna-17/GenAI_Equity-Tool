import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from modules.llm_helper import summarize_live_news, merge_sentiment_and_price, run_rag_query

# Streamlit Page Config 
st.set_page_config(page_title="Equity Research Assistant", layout="wide")
st.title("📊 Equity Research Assistant")
st.markdown("Use this tool to get real-time news analysis, sentiment scoring, and price history for a given ticker.")
st.markdown("---")

# User Input 
companies = st.text_input(
    "1. Enter company ticker or name (comma-separated, e.g., AAPL, MSFT):"
)

# RAG QUESTION INPUT
rag_question = st.text_input(
    "2. Ask a follow-up question about the analyzed news (optional, e.g., What risks were mentioned?):",
    key="rag_q" # Use a unique key
)
st.markdown("---")

# Button Action 
if st.button("Analyze News & Price"):
    if not companies.strip():
        st.warning("Please enter at least one company ticker or name.")
    else:
        tickers = [c.strip() for c in companies.split(",")]
        
        # Add a spinner while working
        with st.spinner("Fetching news, summarizing, indexing for RAG, and plotting data..."):
            
            for ticker in tickers:
                st.header(f"Analysis for {ticker.upper()}")
                st.divider()

                # --- Fetch & summarize news ---
                results = summarize_live_news(ticker)

                if not results:
                    st.warning(f"No recent news found for {ticker}.")
                    continue
                
                
                # NEW: RAG Context Filtering UI 
                all_sources = sorted(list(set(r.get('source') for r in results if r.get('source'))))
                
                st.subheader(f"RAG Context Filters for {ticker.upper()}")
                
                # Multiselect widget for source filtering
                selected_sources = st.multiselect(
                    "Filter RAG Context by News Source:",
                    options=all_sources,
                    default=all_sources, # Default to all sources selected
                    key=f"source_filter_{ticker}"
                )
                st.markdown("💡 Only news from the **selected sources** will be used to answer your follow-up question.")
                st.divider()


                # Display summaries and sentiments 
                st.subheader(f"Latest News Summaries for {ticker.upper()}")
                for r in results:
                    # Apply color coding to the sentiment text
                    sentiment_color = "green" if "Bullish" in r['sentiment'] else ("red" if "Bearish" in r['sentiment'] else "orange")
                    
                    st.markdown(f"**📰 {r['title']}**")
                    st.write(f"**Source:** *{r['source']}*")
                    st.write(f"**Summary:** {r['summary']}")
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}; font-weight:bold;'>{r['sentiment']}</span>", unsafe_allow_html=True)
                    st.markdown(f"[Read Full Article]({r['url']})")
                    st.markdown("---")


                #  RAG QUERY SECTION
                if rag_question.strip():
                    if selected_sources:
                        st.subheader(f"💬 Answer to Follow-up Question: *{rag_question}*")
                        
                        # Call RAG function with the new source filter
                        rag_answer = run_rag_query(results, rag_question, ticker, selected_sources)
                        
                        st.info(rag_answer)
                        st.divider()
                    else:
                        st.warning("You asked a question, but no news sources are selected for the RAG query.")
                
                
                # Merge stock data with sentiment and Plotting 
                merged_df = merge_sentiment_and_price(results, ticker.upper())

                if merged_df.empty:
                    st.warning(f"Unable to fetch stock data for {ticker}.")
                    continue

                # Clean sentiment labels (necessary for color mapping) 
                merged_df["sentiment"] = (
                    merged_df["sentiment"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .replace({"Bull": "Bullish", "Bear": "Bearish"})
                )

                color_map = {"Bullish": "green", "Neutral": "orange", "Bearish": "red"}
                
                # Corrected Plot Section
                st.subheader(f"{ticker.upper()} – Stock Price vs Sentiment (Last 7 Days)")

                #Prepare data for plotting
                last_row = merged_df.iloc[-1]
                last_date = last_row["date"]
                last_score = last_row["sentiment_score"]
                last_sentiment = last_row["sentiment"]
                last_color = color_map.get(last_sentiment, "gray")

                fig, ax = plt.subplots(figsize=(10, 5))

                # 1️⃣ Stock price line (as primary axis)
                ax.plot(
                    merged_df["date"],
                    merged_df["Close"],
                    label="Stock Price",
                    color="blue",
                    linewidth=2,
                    marker='o', 
                    zorder=2 
                )
                ax.set_ylabel("Stock Price", color="blue")
                ax.tick_params(axis='y', labelcolor='blue')

                # 2️⃣ Sentiment overlay (Secondary axis - ONLY plot today's sentiment)
                ax2 = ax.twinx()
                sentiment_dates = [last_date]
                sentiment_scores = [last_score * 100]
                sentiment_colors = [last_color]

                ax2.bar(
                    sentiment_dates,
                    sentiment_scores,
                    color=sentiment_colors,
                    width=0.8, 
                    alpha=0.5, 
                    label=f"Today's Sentiment: {last_sentiment}",
                    zorder=1 
                )
                ax2.set_ylim(-105, 105) 
                ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5) 
                ax2.set_ylabel("Sentiment Score (Max $\\pm 100$)", color="gray")
                ax2.tick_params(axis='y', labelcolor='gray')


                # Final chart polish
                fig.suptitle(f"{ticker.upper()} Price vs Current Sentiment", fontsize=14)
                ax.set_xlabel("Date")
                
                # Combine legends
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')

                # Auto-format date ticks
                fig.autofmt_xdate()
                fig.tight_layout()

                st.pyplot(fig)
                st.divider()
