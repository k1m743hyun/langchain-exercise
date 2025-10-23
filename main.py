from langchain_community.tools import TavilySearchResults

if __name__ == '__main__':
    query = "2025년 애플의 주가 전망에 대해서 분석하세요."

    web_search = TavilySearchResults(max_results=2)

    search_results = web_search.invoke(query)

    for result in search_results:
        print(result)
        print("-" * 100)