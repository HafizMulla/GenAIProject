import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from portfolio import Portfolio
from chains import Chain
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Cold Mail Generator")
    url_input = st.text_input("Enter a URL", value="https://careers.nike.com/lead-machine-learning-engineer-ai-ml-remote-work-option/job/R-38652")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills',[])
                links = portfolio.query_link(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error("An error occured", e)

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout='wide', page_title='Cold Mail Generator', page_icon='A')
    create_streamlit_app(chain, portfolio, clean_text)