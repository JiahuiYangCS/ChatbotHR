import streamlit as st
import os
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"

chat = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

with open("all_documents_index.pkl", 'rb') as f: 
    faiss_index = pickle.load(f)
    
message_history = []
messages = [
    SystemMessage(
        content="You are a Q&A bot and you will answer all the questions that the user has. If you don't know the answer, output 'Sorry, I don't know' .")
]

def predict(input):
    '''Find the k best matched chunks to the queried test. 
    These will be the context over which our bot will try to answer the question.
    The value of k can be adjusted so as to get the embeddings for the best n chunks.'''

    docs = faiss_index.similarity_search(input, K = 6)

    main_content = input + "\n\n"
    for doc in docs:
        main_content += doc.page_content + "\n\n"

    message_history.append({"role": "user", "content": f"{input}"})

    messages.append(HumanMessage(content=main_content))
    ai_response = chat(messages).content
    messages.pop()
    messages.append(HumanMessage(content=input))
    messages.append(AIMessage(content=ai_response))

    message_history.append({"role": "assistant", "content": f"{ai_response}"}) 
    
    # get pairs of msg["content"] from message history, skipping the pre-prompt: here.
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(0, len(message_history)-1, 2)]  # convert to tuples of list
    ai_response = chat(messages).content
    
    message_history.append({"role": "user", "content": f"{input}"})
    message_history.append({"role": "assistant", "content": f"{ai_response}"})
    
    return ai_response

def display_response(response):
    if not isinstance(response, str):
        st.error("Response is not a string.")
        return

    container = st.empty()
    display_text = ""
    for char in response:
        display_text += char
        container.markdown(display_text)
        time.sleep(0.05)
    time.sleep(2)
    container.empty()
    
def main():
    st.title("Chatbot 示例")

    # 用户输入
    user_input = st.text_input("请输入您的消息：", key="user_input")

    
    if st.button("发送"):
        response = predict(user_input)  
        display_response(response)

if __name__ == "__main__":
    main()