from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
# from langchain.chat_models import AzureChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
load_dotenv()
api_key = os.getenv("openai_api_key")
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""

client = OpenAI(api_key=api_key)

search = DuckDuckGoSearchRun()

llm = AzureChatOpenAI(
    openai_api_version="",
    azure_deployment="",
)

# def get_answer(messages):
#     system_message = [{"role": "system", "content": "You are an helpful AI chatbot, that answers questions asked by User."}]
#     messages = system_message + messages
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         messages=messages
#     )
#     return response.choices[0].message.content

prompts = """**You are a helpful and informative chatbot designed to provide accurate and focused answers to questions. Your primary goals are:**

* **Comprehensiveness:** Thoroughly understand the given question and its context to provide the most relevant information possible. 
* **Context-Awareness:** Base your answers **solely** on the provided {context}.  Do not make assumptions, introduce irrelevant details.
* **Multilingual Support:** If a question is asked in Language A i.e Hindi, Urdu Spanish or French, strive to provide a response in the same language.

**Instructions:**

Answer the following Question: {question}

"""


def get_answer(messages):
    new_msg = str(messages[1]["content"])
    # print(new_msg)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(prompts)
    model = ChatOpenAI(model="gpt-4", temperature=0)
    # llm = AzureChatOpenAI(openai_api_version="",azure_deployment="",)
    # languages = language
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )

    data = chain.invoke(new_msg)
    return data

def duck_duck_go(messages):
    new_msg = str(messages[1]["content"])
    data = search.run(new_msg)
    return data

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
