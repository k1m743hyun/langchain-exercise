from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    llm = Ollama(model="gemma:latest")

    #response = llm.invoke("지구의 자전 주기는?")
    #print(response)

    prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")

    chain = prompt | llm

    response = chain.invoke({"input": "지구의 자전 주기는?"})
    print(response)