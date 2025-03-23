import os

from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('GROG_API_KEY'),
            model_name="llama-3.3-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
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
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big.Unable to parse jobs.")

        return res if isinstance(res, list) else [res]


    def write_email(self, job, links):
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

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
