import os
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ['GOOGLE_API_KEY'] = ''

if __name__ == '__main__':
    python_repl = PythonAstREPLTool()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
    )

    agent = initialize_agent(
        [python_repl],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent.run('''
        1부터 10까지의 숫자 중 짝수만 출력하는 Python 코드를 작성하고 실행해주세요.
        그리고 그 결과를 설명해주세요.
    ''')

    print(result)