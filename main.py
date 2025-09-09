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

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    docs = retriever.get_relevant_documents(query)

    # Prompt
    template = '''Answer the question based only on the following
    context:
    {context}
    
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # Chain
    chain = {'context':  retriever | format_docs, 'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()

    # Run
    response = chain.invoke('카카오뱅크의 최근 영업실적을 요약해서 알려주세요.')
    print(response)