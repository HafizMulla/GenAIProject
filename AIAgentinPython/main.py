from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('GROG_API_KEY'),
            model_name="llama-3.3-70b-versatile"
        )
response = llm.invoke("What is the meaning of life")
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
        ("placeholder","{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query":"What is India"})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response.topic)
except Exception as e:
    print("Error parsing response",e,"Raw response -", raw_response)