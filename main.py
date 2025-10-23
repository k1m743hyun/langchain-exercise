import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

os.environ['GOOGLE_API_KEY'] = ''

if __name__ == '__main__':
    loader = PyMuPDFLoader('data/323410_카카오뱅크_2023.pdf')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )

    documents = text_splitter.split_documents(data)

    embedding_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

    # Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=500,
    )

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':7, 'fetch_k':20}
    )

    docs = retriever.get_relevant_documents(query)
    #print(len(docs))

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    compressed_docs = compression_retriever.get_relevant_documents(query)
    print(len(compressed_docs))
    print(compressed_docs)