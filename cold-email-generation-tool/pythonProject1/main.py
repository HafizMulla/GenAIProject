from click import prompt
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uuid
import chromadb

llm = ChatGroq(
    temperature = 0,
    groq_api_key = 'XXXX',
    model_name = "llama-3.3-70b-versatile"
)
# response = llm.invoke('First person to walk over the moon')
# print(response.content)
loader = WebBaseLoader("https://careers.nike.com/lead-machine-learning-engineer-ai-ml-remote-work-option/job/R-38652")
page_data = loader.load().pop().page_content
# print(page_data)

prompt_extract = PromptTemplate.from_template(
    """
    ### SCARPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scrapped text is from the career's page of a website.
    Your job is to extract the job posting and return them in JSON Format containing the
    following keys: role,experience, skills and description.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE): 
    """
)

chain_extract = prompt_extract | llm

res = chain_extract.invoke(input={'page_data':page_data})
# print(res.content)
# print(type(res.content))
json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
# print(json_res)

import pandas as pd
df = pd.read_csv('app/resource/my_portfolio.csv')
# print(df)

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name='portfolio')

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row['Techstack'],
                       metadatas={"links":row['Links']},
                       ids=[str(uuid.uuid4())])

job = json_res
# print(job['skills'])

links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas',[])
# print(links)

prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Hafiz, a bussiness development executive at AtliQ.AtliQ is an AI & Software consulting company.
    Your job is to write a cold email to client regarding the job mentioned above  in fulfilling their needs.
    Also add the most relevant one from the following links to showcase Atliq's portfolio : {link_list}
    Remember you are Hafiz, BDE at AtliQ.
    ### EMAIL (NO PREAMBLE):
    """
)

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description":str(job), "link_list":links})
print(res.content)