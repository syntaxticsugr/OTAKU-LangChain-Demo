from constants import openai_key
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough



llm = OpenAI(api_key=openai_key, temperature=0.7)



about_prompt = PromptTemplate.from_template(
    "Tell me about the Anime Character {name}."
)

dob_prompt = PromptTemplate.from_template(
    "When was {character} born."
)

events_prompt = PromptTemplate.from_template(
    "Tell some major events that occured around the world on {dob}."
)



about_chain = about_prompt | llm | StrOutputParser()

dob_chain = dob_prompt | llm | StrOutputParser()

events_chain = events_prompt | llm | StrOutputParser()



chain = (
    {"character": about_chain}
    | RunnablePassthrough.assign(dob=dob_chain)
    | RunnablePassthrough.assign(events=events_chain)
)



st.title("OTAKU: Langchain Demo With OPENAI API")
st.text(" \n ")

input_text = st.text_input(label="", placeholder="Search For Any Anime Character")
st.text(" \n \n ")

if input_text:
    results = chain.invoke({"name": input_text})

    with st.expander("Character Info"):
        st.write(results["character"])
    
    with st.expander("Birth Date"):
        st.write(results["dob"])

    with st.expander("Events on his/her Birth Date"):
        st.write(results["events"])
