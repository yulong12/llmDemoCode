import datasets
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from smolagents import OpenAIServerModel,WebSearchTool,ToolCallingAgent
from utils.utils import find_files,SemanticRetriever

DS_KEY = "sk-a30bad8dd8e84a3793fad548613df9a3"
DATA_PATH = "data/rag"
TMP_PATH = "/archive/share/cql/aaa/tmp"
EMBEDDING_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/gte-small-zh"
INDEX_DIR = "data/faiss_rag"
SUBSET = -1

directories = ["data"]
data_files = find_files(directories,DATA_PATH)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

if Path(INDEX_DIR).exists():
    vectordb = FAISS.load_local(INDEX_DIR, 
                                embeddings=embedding_model, 
                                allow_dangerous_deserialization=True)
else:
    knowledge_base = datasets.load_dataset("parquet", data_files=data_files, split="train", cache_dir=TMP_PATH) 
    if SUBSET > 0:
        knowledge_base = knowledge_base.select(range(SUBSET))
    source_docs = [
        Document(page_content=doc["content"], metadata={"url": doc["url"]}) for doc in knowledge_base
    ]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH),
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        # import IPython; IPython.embed()
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)
                
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    vectordb.save_local(INDEX_DIR)


model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DS_KEY, 
    flatten_messages_as_text=True,
)

retriever_tool = SemanticRetriever(vectordb)
webserach_tool = WebSearchTool()
agent = ToolCallingAgent(tools=[retriever_tool,webserach_tool], model=model)

while True:
    query = input("Enter your query: ")
    agent_output = agent.run(query)

    print("Response:",agent_output)