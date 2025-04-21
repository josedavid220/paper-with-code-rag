from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chroma_utils import vectorstore
from langchain_core.tools import Tool

from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_google_community import GoogleSearchAPIWrapper

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

output_parser = StrOutputParser()

# Set up prompts and chains
contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt_system = """
    You are an expert in academic papers related to math, AI and computer science.
    You will be provided with context to answer a user's question related to a specific paper that also has a code implementation.
    You should base your answer on both the context from the paper and the code implementation.
    If the user asks where in the code something happens, you should provide the code exactly as it is, without any modifications.
    If the user asks for a modification, you should provide the code with the modification, but also explain what you did and why.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_prompt_system),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4.1-nano")

def get_rag_chain():
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

google_query_system_prompt = """
    Given the user's latest question,
    formulate a concise search query that will retrieve the most relevant external information.
    Return ONLY the search query.
"""

google_query_prompt = ChatPromptTemplate.from_messages([
    ("system", google_query_system_prompt),
    ("human", "{input}")
])

google_query_chain = google_query_prompt | llm | StrOutputParser()

search = GoogleSearchAPIWrapper(k=3)
tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )

google_system_prompt = """
    You are an expert in academic papers related to math, AI and computer science. 
    Use the following external information, which was obtained from a Google search,
    plus the chat history, to answer the user's question.
"""
google_prompt = ChatPromptTemplate.from_messages([
    ("system", google_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("system", "External Info: {search_results}"),
    ("human", "{query}")
])
google_chain = google_prompt | llm | StrOutputParser()

route_system = (
    "Decide whether the user's query should be answered using the indexed documents (RAG) "
    "or via a live Google search."
)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", route_system),
    ("human", "{query}")
])

class RouteOutput(TypedDict):
    destination: Literal["rag", "google"]

route_chain = route_prompt | llm.with_structured_output(RouteOutput)

class State(TypedDict):
    query: str
    chat_history: list
    destination: RouteOutput
    answer: str

async def route_query(state: State, config):
    decision = await route_chain.ainvoke({"query": state["query"]})
    return {"destination": decision}


async def run_rag(state: State, config):
    rag_chain = get_rag_chain()
    result = await rag_chain.ainvoke({
        "input": state["query"],
        "chat_history": state["chat_history"],
    })
    return {"answer": result["answer"]}


async def run_google(state: State, config):
    search_query = await google_query_chain.ainvoke({
        "input": state["query"],
        "chat_history": state["chat_history"],
    })
    
    results = tool.run(search_query)
    
    answer = await google_chain.ainvoke({
        "search_results": results,
        "query": state["query"],
        "chat_history": state["chat_history"],
    })
    return {"answer": answer}


def select_node(state: State) -> Literal["run_rag", "run_google"]:
    return "run_google" if state["destination"]["destination"] == "google" else "run_rag"

graph = StateGraph(State)
graph.add_node("route_query", route_query)
graph.add_node("run_rag", run_rag)
graph.add_node("run_google", run_google)

graph.add_edge(START, "route_query")
graph.add_conditional_edges("route_query", select_node)
graph.add_edge("run_rag", END)
graph.add_edge("run_google", END)

agent_app = graph.compile()