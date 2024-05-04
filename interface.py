import gradio as gr
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent

load_dotenv()

agent = create_csv_agent(ChatOpenAI(model="gpt-3.5-turbo", temperature=0), 
                         "/Users/MSI/Desktop/Turkcell_Hackathon/Chatbot_Pasaj_Turkcell/data/EbayPcLaptopsAndNetbooksClean.csv", 
                         verbose=True)

# Define function for chatbot conversation
def chatbot_conversation(request):
    response = agent.invoke(request)

    return response["output"]
        

# Define Gradio interface for chatbot
chatbot_interface = gr.Interface(
    fn=chatbot_conversation,
    inputs=gr.Textbox(label="Enter your request"),
    outputs=gr.Textbox(label="Laptops"),
    title="Pasaj Chatbot",
    description="Ask about laptops available in our store!",
)

chatbot_interface.launch(share=True)