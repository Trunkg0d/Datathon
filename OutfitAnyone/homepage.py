
import streamlit as st
from PIL import Image
import os

link_fittingRoom ="https://dcf8-14-161-23-204.ngrok-free.app/?fbclid=IwAR0NXqACK0VJX0Eny-iaClssqMngxgIFoBXjU6XFuG0yoW5jilNtBA4gvVI"
link_chatbot = "https://9ede-115-73-213-165.ngrok-free.app/?fbclid=IwAR3XBe74e6J9WGtz8nnEMnMnYuSS6PF2QIrViasuutnjEz8PfCPWrDycrLw"
st.markdown("<h1 style='text-align:center'>Image Process App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size: 20px;'>Welcome to our Application</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size: 20px;'>This app contains two feature</p>", unsafe_allow_html=True)
st.markdown(f"""
<div style='margin:auto;text-align:center; display: flex; flex-direction: row; justify-content:space-around;flex-wrap:wrap '>
<div style='display:flex; flex-direction: column; text-align: center'><img src="#" width="200px" height="200px" style='padding:10px; margin: 10px'>NAME</div>
<div style='display:flex; flex-direction: column; text-align: center'><img src="#" width="200px" height="200px" style='padding:10px; margin: 10px'>NAME</div>
<div style='display:flex; flex-direction: column; text-align: center'><img src="#" width="200px" height="200px" style='padding:10px; margin: 10px'>NAME</div>
<div style='display:flex; flex-direction: column; text-align: center'><img src="#" width="200px" height="200px" style='padding:10px; margin: 10px'>NAME</div>
<div style='display:flex; flex-direction: column; text-align: center'><img src="#" width="200px" height="200px" style='padding:10px; margin: 10px'>NAME</div>
</div>
""", unsafe_allow_html=True)
# Display the Markdown content
with st.sidebar:
    fittingRoom = st.link_button(label="Virtual Fitting Room", url=link_fittingRoom)
    chatBot = st.link_button(label="Chatbot",url=link_chatbot)

if fittingRoom:
    st.experimental_set_query_params(route="/")
if chatBot:
    st.experimental_set_query_params(route="/")

query_params = st.experimental_get_query_params()

# Retrieve the value of the "route" parameter
route = query_params.get("route", ["/"])[0]
