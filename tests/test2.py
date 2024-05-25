from constants import openai_key
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



llm = OpenAI(api_key=openai_key, temperature=0.7)



prompt1 = PromptTemplate.from_template(
    "Tell me about the Anime Character {name}."
)

prompt2 = PromptTemplate.from_template(
    "When was {character} born."
)

prompt3 = PromptTemplate.from_template(
    "Tell some major events that occured around the world on {dob}."
)



chain = (
    prompt1
    | llm
    | StrOutputParser()
    | (lambda input: {"character": input})
    | prompt2
    | llm
    | StrOutputParser()
    | (lambda input: {"dob": input})
    | prompt3
    | llm
    | StrOutputParser()
)



st.title("OTAKU: Langchain Demo With OPENAI API")
input_text = st.text_input("Search For Any Anime Character")

if input_text:
    st.write(
        chain.invoke({"name": input_text})
    )
