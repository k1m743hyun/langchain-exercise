from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=300,
        api_key=""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유익한 AI 조수입니다. 한국의 역사와 문화에 대해 잘 알고 있습니다."),
        ("human", "{question}")
    ])

    chain = prompt | llm

    questions = [
        "한글의 창제 원리는 무엇인가요?",
        "김치의 역사와 문화적 중요선에 대해 설명해주세요.",
        "조선시대의 과거 제도에 대해 간단히 설명해주세요."
    ]

    for question in questions:
        response = chain.invoke({"question": question})
        print(f"질문: {question}")
        print(f"답변: {response}\n")