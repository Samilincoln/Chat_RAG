import streamlit as st
from decouple import config
import asyncio
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from scraper.scraper import process_urls
from embedding.vector_store import initialize_vector_store





#Groq API Key
groq_api = config("GROQ_API_KEY")

#Initializing LLM with memory
llm = ChatGroq(model="llama-3.2-1b-preview", groq_api_key=groq_api, temperature=0)

# ✅ System Prompt with history
system_prompt = """
You are a conversational AI assistant capable of answering user queries.

- **For factual questions**, use **only** the retrieved context below. If the answer is not in the context, say **"I don't know."**  
- **For general conversational inputs** (e.g., greetings like hi, hello and other small talks), respond naturally.  
- **Do not make assumptions or generate false information.**  

**Chat History:**  
{history}

**Context:**  
{context}  

Now, answer concisely.
"""

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

#Streamlit
st.title("Botty 1.0 🤖")

#url inputs
urls = st.text_area("Enter URLs (one per line)")
run_scraper = st.button("Run Scraper", disabled=not urls.strip())

#Sessions & states
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "history" not in st.session_state:
    st.session_state.history = ""  # Stores past Q&A for memory
if "scraping_done" not in st.session_state:
    st.session_state.scraping_done = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

#Run scraper
if run_scraper:
    st.write("Fetching and processing URLs... This may take a while.")
    split_docs = run_asyncio_coroutine(process_urls(urls.split("\n")))
    st.session_state.vector_store = initialize_vector_store(split_docs)
    st.session_state.scraping_done = True
    st.success("Scraping and processing completed!")



#Ensuring chat only enables after scraping
if not st.session_state.scraping_done:
    st.warning("Scrape some data first to enable chat!")
else:
    st.write("### Chat With Botty 💬")

    #Display chat history
    for message in st.session_state.messages:
        role, text = message["role"], message["text"]
        with st.chat_message(role):
            st.write(text)

    #Takes in user Input
    user_query = st.chat_input("Ask a question...")

    if user_query:
        #Show user's message
        st.session_state.messages.append({"role": "user", "text": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        #Retrieval
        retriever = st. session_state.vector_store.as_retriever(search_kwargs={'k': 3})
        scraper_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        llm_chain = create_retrieval_chain(retriever, scraper_chain)

        #Memory update
        history_text = "\n".join(
            [f"User: {msg['text']}" if msg["role"] == "user" else f"AI: {msg['text']}" for msg in st.session_state.messages]
        )
        st.session_state.history = history_text

        #Context retrieve
        retrieved_docs = retriever.invoke(user_query)
        response = llm_chain.invoke({"input": user_query, "history": st.session_state.history})

        answer = response["answer"]
        source_url = retrieved_docs[0].metadata.get("source", "Unknown") if retrieved_docs else "No source found"

        #Format response with memory tracking
        formatted_response = f"**Answer:** {answer}\n\n**Source:** {source_url}"

        #Store and display bot response
        st.session_state.messages.append({"role": "assistant", "text": formatted_response})
        with st.chat_message("assistant"):
            st.write(formatted_response)
