import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

llm = OpenAI(api_key=openai_key, temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're the world's biggest otaku and a total weeaboo."),
    ("user", "Tell me about the Anime Character {name}.")
])

chain = prompt | llm | StrOutputParser()

st.title("OTAKU: Langchain Demo With OPENAI API")
input_text = st.text_input("Search For Any Anime Character")

if input_text:
    st.write(
        chain.invoke({"name": input_text})
    )
