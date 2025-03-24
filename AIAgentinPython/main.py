import os

from langchain_groq import ChatGroq
from numpy.f2py.crackfortran import verbose
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_use:list[str]

llm = ChatGroq(
    temperature = 0,
    groq_api_key = os.getenv('GROG_API_KEY'),
    model_name = "llama-3.3-70b-versatile"
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ('placeholder',"{chat_history}"),
        ('human',"{query}"),
        ('placeholder',"{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
query=input("What can I help you research?")
raw_response = agent_executor.invoke({"query":query})
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw response - ",raw_response)