from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import chain

async def process_urls(urls):
    #Scraping the URLs using AsyncChromiumLoader
    loader = AsyncChromiumLoader(urls)
    docs = await loader.aload()

    #Transforming the scraped HTML to text 
    text_transformer = Html2TextTransformer()
    transformed_docs = text_transformer.transform_documents(docs)

    #Chunking the transformed text into smaller pieces for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    split_docs_nested = [text_splitter.split_documents([doc]) for doc in transformed_docs]
    split_docs = list(chain.from_iterable(split_docs_nested))

    #Preserving the source URL
    split_docs = []
    for doc_list, original_doc in zip(split_docs_nested, transformed_docs):
        for chunk in doc_list:
            chunk.metadata["source"] = original_doc.metadata.get("source", "Unknown")  # Preserve URL
            split_docs.append(chunk)

    return split_docs