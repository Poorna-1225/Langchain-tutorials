import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


##prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are an helpful assistant. Please respond to the question asked'),
        ('user','Question:{question}')
    ]
)

##streamlit framework
st.title("Langchain Demo with Gemma2:2b model")
input_text =  st.text_input("Enter your question")

## calling ollama gemma:2b model
llm = Ollama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    llm_response = chain.invoke({'question':input_text})
    st.write(llm_response)


"""
This code sets up a simple chatbot interface using Streamlit and the Ollama language model. Let's break down the code step by step:

**1. Environment Setup**

* `import os`: Imports the `os` module for interacting with the operating system.
* `from dotenv import load_dotenv`: Imports the `load_dotenv` function from the `python-dotenv` library. This library allows you to load environment variables from a `.env` file.
* `load_dotenv()`: Loads the environment variables from the `.env` file into the current environment.
* `os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')`: Sets the `LANGCHAIN_API_KEY` environment variable with the value from the `.env` file. This key is likely used for Langchain's tracing and logging features.
* `os.environ['LANGCHAIN_TRACING_V2'] = 'true'`: Enables Langchain's tracing feature, which helps you monitor and debug your Langchain applications.
* `os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')`:  Sets the `LANGCHAIN_PROJECT` environment variable, likely used for organizing your Langchain projects.

**2. Importing Libraries**

* `from langchain_community.llms import Ollama`: Imports the `Ollama` class from the `langchain_community.llms` module. This class provides an interface for interacting with the Ollama language model.
* `import streamlit as st`: Imports the `streamlit` library, which is used to create interactive web applications.
* `from langchain.prompts import ChatPromptTemplate`: Imports the `ChatPromptTemplate` class from `langchain.prompts`. This class helps you define prompts for your language model in a structured way.
* `from langchain.output_parsers import StrOutputParser`: Imports the `StrOutputParser` class from `langchain.output_parsers`. This class is used to parse the output from the language model into a string.

**3. Defining the Prompt Template**

* `prompt = ChatPromptTemplate.from_messages(...)`: Creates a `ChatPromptTemplate` object. This template defines the structure of the conversation with the language model.
    * `('system', 'You are a helpful assistant. Please respond to the question asked')`: This message sets the "system" role in the conversation, instructing the language model to act as a helpful assistant.
    * `('user', 'Question:{question}')`: This message defines the "user" role and includes a placeholder `{question}` where the user's actual question will be inserted.

**4. Building the Streamlit App**

* `st.title("Langchain Demo with Gemma2:2b model")`: Sets the title of the Streamlit app.
* `input_text = st.text_input("Enter your question")`: Creates a text input field in the app where the user can enter their question. The entered text is stored in the `input_text` variable.

**5. Initializing the Language Model and Chain**

* `llm = Ollama(model="gemma:2b")`: Initializes an `Ollama` object with the specified model (`gemma:2b`).
* `output_parser = StrOutputParser()`: Creates a `StrOutputParser` object to parse the model's output.
* `chain = prompt | llm | output_parser`: Creates a chain that connects the prompt template, the language model, and the output parser. This chain defines the flow of data: the user's question is inserted into the prompt, the prompt is sent to the language model, and the model's output is parsed into a string.

**6. Running the Chatbot**

* `if input_text:`: Checks if the user has entered any text in the input field.
    * `llm_response = chain.invoke({'question': input_text})`: If the user has entered text, it invokes the chain with the user's question. The `invoke` method runs the chain and returns the parsed output from the language model.
    * `st.write(llm_response)`: Displays the language model's response in the Streamlit app.

This code effectively demonstrates how to build a simple chatbot interface using Langchain and Streamlit. It leverages the Ollama language model, defines a clear prompt structure, and uses Streamlit to create an interactive user experience.


"""