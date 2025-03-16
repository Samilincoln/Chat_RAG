import streamlit as st
from decouple import config
import asyncio
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from scraper.scraper import process_urls
from embedding.vector_store import initialize_vector_store, clear_chroma_db
from conversation.talks import clean_input, small_talks

#Clearing ChromaDB at startup to clean up any previous data
clear_chroma_db()




#Groq API Key
groq_api = config("GROQ_API_KEY")

#Initializing LLM with memory
llm = ChatGroq(model="llama-3.2-1b-preview", groq_api_key=groq_api, temperature=0)



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

st.title("WebGPT 1.0 🤖")

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
    st.write("### Chat With WebGPT 💬")

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
            # ✅ Setup retriever (with a similarity threshold or top-k retrieval)
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={'k': 5}
            )

            # ✅ Retrieve context
            retrieved_docs = retriever.invoke(user_query_cleaned)
            retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

            # ✅ Define Langchain PromptTemplate properly
            system_prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are WebGPT, an AI assistant for question-answering tasks that **only answers questions based on the provided context**.

                - If the answer is **not** found in the Context, reply with: "I can't find your request in the provided context."
                - If the question is **unrelated** to the Context, reply with: "I can't answer that."
                - **Do not** use external knowledge, assumptions, or filler responses. Stick to the context provided.
                - Keep responses clear, concise, and relevant to the user’s query.

                Context:
                {context}

                Now, answer the user's question:
                {input}
                """
            )

            # ✅ Generate prompt with retrieved context & user query
            final_prompt = system_prompt_template.format(
                context=retrieved_text,
                input=user_query_cleaned
            )

            # ✅ Create chains (ensure the prompt is correct)
            scraper_chain = create_stuff_documents_chain(llm=llm, prompt=system_prompt_template)
            llm_chain = create_retrieval_chain(retriever, scraper_chain)

            # ✅ Process response and source
            if retrieved_docs:
                try:
                    response_data = llm_chain.invoke({"context": retrieved_text, "input": user_query_cleaned})
                    response = response_data.get("answer", "").strip()
                    source_url = retrieved_docs[0].metadata.get("source", "Unknown")

                    # Fallback if response is still empty
                    if not response:
                        response = "I can't find your request in the provided context."
                        source_url = "No source found"
                        
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
                    source_url = "Error"

            else:
                response = "I can't find your request in the provided context."
                source_url = "No source found"

            # ✅ Track history & update session state
            history_text = "\n".join(
                [f"User: {msg['text']}" if msg["role"] == "user" else f"AI: {msg['text']}" for msg in st.session_state.messages]
            )
            st.session_state.history = history_text

        # ✅ Format and display response
        formatted_response = f"**Answer:** {response}"
        if response != "I can't find your request in the provided context." and source_url:
            formatted_response += f"\n\n**Source:** {source_url}"

        st.session_state.messages.append({"role": "assistant", "text": formatted_response})
        with st.chat_message("assistant"):
            st.write(formatted_response)

