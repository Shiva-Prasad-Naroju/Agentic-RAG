
# THis python code file is the error free and updated code version of Agentic_Rag.ipynb

## Agentic RAG
import os
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Global LLMs (optimized)
# -------------------------------
llm_big = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=256, api_key=GROQ_API_KEY)
llm_small = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=50, api_key=GROQ_API_KEY)

# -------------------------------
# Load Docs
# -------------------------------
urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]

docs=[WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
langgraph_vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings,
)

langgraph_retriever = langgraph_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
langgraph_retriever.invoke("What is mean by Langgraph? Why it is used?")[0]

# Retriever tool
from langchain.tools.retriever import create_retriever_tool
langgraph_retriever_tool=create_retriever_tool(
    langgraph_retriever,
    "retriever_vector_db_blog",
    "Search and run information about Langgraph"
)

# Langchain Blogs
langchain_urls=[
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

docs=[WebBaseLoader(url).load() for url in langchain_urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)

langchain_vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

langchain_retriever = langchain_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

langchain_retriever_tool=create_retriever_tool(
    langchain_retriever,
    "retriever_vector_langchain_blog",
    "Search and run information about Langchain"
)

tools=[langgraph_retriever_tool,langchain_retriever_tool]

# -------------------------------
# Agent State
# -------------------------------
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------------------
# Agent Node
# -------------------------------
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    """
    print("\n---CALL AGENT---")
    messages = state["messages"]

    model = llm_small

    # Binding the tools with llm
    model = model.bind_tools(tools)
    print("Agent model initialized and tools bound.")

    response = model.invoke(messages)
    print("Agent generated a response.")

    return {"messages": [response]}

# -------------------------------
# Grader Node
# -------------------------------
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("\n---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = llm_small
    
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    print("Running relevance check with retrieved documents...")
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

# -------------------------------
# Generate Node
# -------------------------------
def generate(state):
    """
    Generate answer
    """
    print("\n---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")

    llm = llm_big

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    print("Generating final response from retrieved documents...")
    response = rag_chain.invoke({"context": docs, "question": question})
    print("Response generated successfully.")

    return {"messages": [response]}

# -------------------------------
# Rewrite Node
# -------------------------------
def rewrite(state):
    """
    Transform the query to produce a better question.
    """
    print("\n---TRANSFORM QUERY---")

    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    model = llm_small
    print("Rewriting the query for better retrieval...")
    response = model.invoke(msg)
    print("Query rewritten successfully.")

    return {"messages": [response]}

# -------------------------------
# LangGraph Workflow
# -------------------------------
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
retrieve=ToolNode([langgraph_retriever_tool,langchain_retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

# For Jupyter:
# from IPython.display import Image, display
# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# Example queries
result = graph.invoke({"messages": [HumanMessage(content="What is LangChain?")]})
print(result["messages"][-1].content)
