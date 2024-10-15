from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key  = os.getenv('GROQ_API_KEY')
model = ChatGroq(model = ' Gemma2-9b-It', groq_api_key = groq_api_key)


#Create a prompt template

System_template = "Translate the following into{language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', System_template),
        ('user','{text}')
    ]
)

# Create an output parser
output_parser = StrOutputParser()

#create the chain
chain =  prompt_template|model|output_parser

#app definition
app = FastAPI(title="Langchain Server",
              version= "1.0",
              description="A simple server for Langchain Runnabale Interfaces")

add_routes(
    app,
    chain,
    path = "/chain"

)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = '127.0.0.1', port = 8000)