from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(model="bge-m3:lastest")

reasoning_llm = ChatOllama(model="deepseek-r1:7b", stop=["</think>"], streaming=True)

answer_llm = ChatOllama(model="exaone3.5", temperature=0, streaming=True)
