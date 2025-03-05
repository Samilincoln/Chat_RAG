import streamlit as st
from decouple import config
import asyncio
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from scraper.scraper import process_urls
from embedding.vector_store import initialize_vector_store, clear_chroma_db
from conversation.talks import clean_input, small_talks

#Clearing ChromaDB at startup to clean up any previous data
clear_chroma_db()




#Groq API Key
groq_api = config("GROQ_API_KEY")

#Initializing LLM with memory
llm = ChatGroq(model="llama-3.2-1b-preview", groq_api_key=groq_api, temperature=0)

# ✅ System Prompt with history
system_prompt = (
"""
You are Botty an AI assistant that answers questions **strictly based on the retrieved context**.

- **If the answer is found in the context, respond concisely.**  
- **If the answer is NOT in the context, reply ONLY with: "I can't find your request in the provided context."**   
- **If the question is unrelated to the provided context, reply ONLY with: "I can't answer that."**  
- **DO NOT use external knowledge or assumptions.**  

Context:  
{context}  

Now, respond accordingly.
"""
)

#Chat Prompt
prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

#Ensure proper asyncio handling for Windows
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

#Async helper function
def run_asyncio_coroutine(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

import streamlit as st

st.title("Botty 1.0 🤖")

# URL inputs
urls = st.text_area("Enter URLs (one per line)")
run_scraper = st.button("Run Scraper", disabled=not urls.strip())

# Sessions & states
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "history" not in st.session_state:
    st.session_state.history = ""  # Stores past Q&A for memory
if "scraping_done" not in st.session_state:
    st.session_state.scraping_done = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Run scraper
if run_scraper:
    st.write("Fetching and processing URLs... This may take a while.")
    split_docs = run_asyncio_coroutine(process_urls(urls.split("\n")))
    st.session_state.vector_store = initialize_vector_store(split_docs)
    st.session_state.scraping_done = True
    st.success("Scraping and processing completed!")

# Ensuring chat only enables after scraping
if not st.session_state.scraping_done:
    st.warning("Scrape some data first to enable chat!")
else:
    st.write("### Chat With Botty 💬")

    # Display chat history
    for message in st.session_state.messages:
        role, text = message["role"], message["text"]
        with st.chat_message(role):
            st.write(text)

    # Takes in user input
    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.messages.append({"role": "user", "text": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        user_query_cleaned = clean_input(user_query)
        response = "" # Default value for response
        source_url = ""  # Default value for source url

        # Check for small talk responses
        if user_query_cleaned in small_talks:
            response = small_talks[user_query_cleaned]
            source_url = "Knowledge base"  # Small talk comes from the knowledge base
            
        else:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 5})
            scraper_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            llm_chain = create_retrieval_chain(retriever, scraper_chain)

            # Retrieve context
            retrieved_docs = retriever.invoke(user_query_cleaned)
            #retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved docs into a single string


            if retrieved_docs:
                response = llm_chain.invoke({"input": user_query_cleaned})["answer"]  
                source_url = retrieved_docs[0].metadata.get("source", "Unknown")
                
                # Ensuring the response is not empty
                if not response.strip():
                    response = "I can't find your request in the provided context."
                    source_url = "No source found"  # No source if no real answer
            else:
                response = "I can't find your request in the provided context."
                source_url = ""  # No misleading source URL

            # ✅ Memory Tracking & Response Formatting
            history_text = "\n".join(
                [f"User: {msg['text']}" if msg["role"] == "user" else f"AI: {msg['text']}" for msg in st.session_state.messages]
            )
            st.session_state.history = history_text  # Update history

        formatted_response = f"**Answer:** {response}"
        if response != "I can't find your request in the provided context." and source_url:
            formatted_response += f"\n\n**Source:** {source_url}"

        # ✅ Append formatted response
        st.session_state.messages.append({"role": "assistant", "text": formatted_response})
        with st.chat_message("assistant"):
            st.write(formatted_response)

