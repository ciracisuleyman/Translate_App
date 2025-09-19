
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

# Bu yorum: uygulama çeviri yapıyor

# Ortam değişkenini yükle (.env içinde GOOGLE_API_KEY olmalı)
load_dotenv()

# Gemini modelini başlat
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

system_prompt="Translate the following from English to {language}"
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt), ("user", "{text}")
    ]
)

parser=StrOutputParser()
chain=prompt_template | model | parser

app=FastAPI(
    title="Translator App!",
    version="1.0.0",
    description="Translation Chat Bot",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes


# Ortam değişkenini yükle (.env içinde GOOGLE_API_KEY olmalı)
load_dotenv()

# Gemini modelini başlat
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

system_prompt="Translate the following from English to {language}"
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt), ("user", "{text}")
    ]
)
 
    #çıktı parçalara bölünür
parser=StrOutputParser()
chain=prompt_template | model | parser

app=FastAPI(
    title="Translator App!",
    version="1.0.0",
    description="Translation Chat Bot",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

