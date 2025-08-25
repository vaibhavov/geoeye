from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

llm = ChatOllama(model="mistral", temperature=0.2, base_url="http://localhost:11434")

resp = llm.invoke([HumanMessage(content="Summarize NDVI, NDRE, NDWI in one paragraph.")])
print(resp.content)
