import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_google_genai import ChatGoogleGenerativeAI

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
    #print(len(documents))

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

    # Retrieval

    # 단일 문서 검색 - 가장 유사도가 높은 문장을 하나만 추출
    '''
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
    '''

    # MMR(Maximal Marginal Relevance) 검색
    '''
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 50}
    )
    '''

    # MMR(Maximal Marginal Relevance) 검색 - lambda_mult 작을수록 더 다양하게 추출
    '''
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'lambda_mult': 0.15}
    )
    '''

    # 유사도 점수 임계값 기반 검색 - 기준 스코어 이상인 문서를 대상으로 추출
    '''
    retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={'score_threshold': 0.3}
    )
    '''

    # 메타데이터 필터링을 사용한 검색 - 문서 객체의 metadata를 이용한 필터링
    '''
    retriever = vectorstore.as_retriever(
        search_kwargs={'filter': {'format': 'PDF 1.4'}}
    )
    '''

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'lambda_mult': 0.15}
    )

    docs = retriever.get_relevant_documents(query)
    '''
    print(len(docs))
    print(docs[0])
    '''

    # Prompt
    template = '''Answer the question based only on the following
    context:
    {context}
    
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)

    # Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=500,
    )

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # Chain
    chain = prompt | llm | StrOutputParser()

    # Run
    response = chain.invoke({'context': (format_docs(docs)), 'question': query})
    print(response)