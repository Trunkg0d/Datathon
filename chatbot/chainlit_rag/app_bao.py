#import required libraries
from image import *
import re
import urllib.request 
import webbrowser
import time
from PIL import Image 
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter ,  CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
 SystemMessagePromptTemplate,
 HumanMessagePromptTemplate)
from langchain.llms import HuggingFaceHub
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from chainlit import run_sync
from cohere import Client
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from getpass import getpass

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

HUGGINGFACEHUB_API_TOKEN = getpass()

import os
from configparser import ConfigParser
env_config = ConfigParser()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
 assert parser.read(location), f"Could not read config {location}"
#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = api_key
api_key = env_config.get("hgface", "api_key").strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

text_splitter = CharacterTextSplitter(
    separator="[PRODUCT]",
    chunk_size=1,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
)

text_splitter_2 = CharacterTextSplitter(
    separator="[CRITERIA]",
    chunk_size=1,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
)

system_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{You are and assistant fashion bot of a virtual fitting room, your job is help the person to get there right outfit and try on clothes virtually. If you don't have that piece of cloths, just sorry the person do not show the product and ask them for more
request}
----------------
{question}

Your response:
"""
human_template = """I'm finding some clothes/outfit on your context provided"""
messages = [SystemMessagePromptTemplate.from_template(system_template),HumanMessagePromptTemplate.from_template("{question}")]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def has_recommend(question):
    #special_context = ['recommend' , 'score']
    special_context = ['recommend']
    for item in special_context:
        if item in question:
            return True
    return False


def has_scoring(question):
    special_context = ['score']
    for item in special_context:
        if item in question:
            return True
    return False

async def upload_and_rating():
    file = await cl.AskFileMessage(
        content="Please upload your image", accept = {'image/png': ['.png' ,'.jpg' ,'.jpeg']}
    ).send()
    dataBytesIO = io.BytesIO(file[0].content)
    raw_image = Image.open(dataBytesIO).convert('RGB')
    text = "a person is wearing "
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs )
    image_caption = processor.decode(out[0], skip_special_tokens=True , max_length = 100)  
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    input = "Given " + image_caption + " what is your recommendation outift for this person  " 
    response = await chain.acall(input, callbacks=[cb])
    await cl.Message(content= response["result"]).send()    
    
    
# async def scoring_outfit () :
#     file1 = await cl.AskFileMessage(
#         content="Please upload your image", accept = {'image/png': ['.png' ,'.jpg' ,'.jpeg']}
#     ).send()
#     file2 = await cl.AskFileMessage(
#         content="Please upload your image", accept = {'image/png': ['.png' ,'.jpg' ,'.jpeg']}
#     ).send()
    
#     dataBytesIO = io.BytesIO(file1[0].content)
#     raw_image = Image.open(dataBytesIO).convert('RGB')
#     text = "a person is wearing "
#     inputs = processor(raw_image, text, return_tensors="pt")
#     out = model.generate(**inputs )
#     image_caption = processor.decode(out[0], skip_special_tokens=True , max_length = 100)
    
async def show_image(answer , image_dict):
    # pattern = r'\./image.*?\.png'
    # req = answer + '. SYSTEM CALL: Only show the image path! Provide exactly the image path based on the image name, do not allow to edit the image path!'
    # chain = cl.user_session.get("chain") 
    # cb = cl.AsyncLangchainCallbackHandler(
    # stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    # cb.answer_reached = True
    # response = await chain.acall(req, callbacks=[cb])
    # new_answer = response["result"]
    # matches = re.findall(pattern, new_answer)
    actions = [
        cl.Action(name="rating_button", value="example_value", description="Click me!"),
        cl.Action(name="enter_virtual_fittingroom", value="example_value", description="Click me!")
    ]
    for item in image_dict:
        if item[5][-4:] == '.png':
            elements = []
            elements.append(cl.Image(name="image1", display="inline", path=item))
            answer = f"Product infomation:\nProduct name: {item[0]}\nProduct description: {item[1]}\nProduct color: {item[2]}\nProduct price: {item[3]}"
            await cl.Message(content=answer,elements=elements, actions=actions).send()    
    # )

@cl.action_callback("rating_button")
async def on_action(action):
    await action.remove()
    await cl.Message(content=f"Your outfit score based on my provided knowledge: 10").send()
    time.sleep(2)
    webbrowser.open('http://example.com')

@cl.action_callback("enter_virtual_fittingroom")
async def on_action(action):
    await action.remove()
    await cl.Message(content=f"Now you can try on your outfit! Have a nice experimence").send()
    webbrowser.open_new_tab('https://courses.fit.hcmus.edu.vn/login/index.php') 

async def show_button():
    actions = [
        cl.Action(name="rating_button", value="example_value", description="Click me!"),
        cl.Action(name="enter_virtual_fittingroom", value="example_value", description="Click me!")
    ]

    await cl.Message(content="You can use these function for more experimence:", actions=actions).send()

#Decorator to react to the user websocket connection event.
@cl.on_chat_start
async def init():
    files = None
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin!",
        accept=["application/pdf"],
        ).send()
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`…")
    await msg.send()
    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)
    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    # Create a Chroma vector store
    # model_id = "BAAI/llm-embedder"
    model_id = "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
    model_kwargs = {"device":"cpu"})
    #
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k=5
    # Store the embeddings in the user session
    cl.user_session.set("embeddings", embeddings)
    docsearch = await cl.make_async(Qdrant.from_texts)(
    texts, embeddings,location=":memory:", metadatas=metadatas
    )
    repo_id = 'HuggingFaceH4/zephyr-7b-beta'
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 1000}
    )
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    #Hybrid Search
    qdrant_retriever = docsearch.as_retriever(search_kwargs={"k":10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,qdrant_retriever],
    weights=[0.5,0.5])
    #Cohere Reranker
    #
    compressor = CohereRerank(client=Client(api_key=os.getenv("COHERE_API_KEY")),user_agent='langchain')
    #
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
    base_retriever=ensemble_retriever,
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    )
    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()
    #store the chain as long as the user session is active
    cl.user_session.set("chain", chain)

@cl.on_message
async def process_response(res:cl.Message):
    # retrieve the retrieval chain initialized for the current session
    chain = cl.user_session.get("chain") 
    # Chinlit callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    print("in retrieval QA")
    print(f"res : {res.content}")
    new_question = res.content + ". Please don't give me the image path! If you have the product, only show the name"
    response = await chain.acall(new_question, callbacks=[cb])
    print(f"response: {response}")
    answer = response["result"] #quan trong dung de lay cau tra loi



    # # #UPLOAD IMAGE
    # file = await cl.AskFileMessage(
    #     content="Please upload your image", accept = {'image/png': ['.png' ,'.jpg' ,'.jpeg']}
    #   ).send()
    # dataBytesIO = io.BytesIO(file[0].content)
    # raw_image = Image.open(dataBytesIO).convert('RGB')
    # text = "a person is wearing "
    # inputs = processor(raw_image, text, return_tensors="pt")
    # out = model.generate(**inputs )
    # image_caption = processor.decode(out[0], skip_special_tokens=True , max_length = 100)
    # input = "Given " + image_caption + " what is your recommendation outift for this person  " 
    
    # #GENERATE RESPONSE
    # response = await chain.acall(res.content, callbacks=[cb])
    # # response = await chain.acall(input, callbacks=[cb])


    #Retrieve source document
    sources = response["source_documents"]
    source_elements = []
    image_dict = []
    for item in sources:
        if item.metadata['relevance_score']>0.5:
            item_data = []
            pattern = r"\[PRODUCT NAME\]: (.+?);"
            match = re.search(pattern, item.page_content)
            product_name = match.group(1)
            item_data.append(product_name)

            pattern = r"\[DESCRIPTION\]: (.+?);"
            match = re.search(pattern, item.page_content)
            product_description = match.group(1)
            item_data.append(product_description)

            pattern = r"\[COLOR\]: (.+?);"
            match = re.search(pattern, item.page_content)
            color = match.group(1)
            item_data.append(color)

            pattern = r"\[PRICE\]: (.+?);"
            match = re.search(pattern, item.page_content)
            price = match.group(1)
            item_data.append(price)

            path = item.page_content.split("[PATH]:")[-1].replace (" " , "").strip().replace ("\n" , "")
            item_data.append(path)

            image_dict.append(item_data)
            print(image_dict)
    # image_path = sources[0].page_content.split("[PATH]:")[-1].replace (" " , "").strip().replace ("\n" , "")
    # best_score = sources[0].metadata['relevance_score']
    # print ("image url" , image_path )
    metadatas = cl.user_session.get("metadatas")

    if cb.has_streamed_final_answer:
        await cb.final_stream.update()
    else:
        print('Here: \n', answer)
        print('Question: ', res.content)
        
        if has_scoring (res.content) :
            print ("Has scoring")
            # run_sync(scoring_outfit())
            cl.user_session.set("chain", chain)
        if has_recommend(res.content):
            print ("Has recommend")
            run_sync(upload_and_rating())
            cl.user_session.set("chain", chain)
        if len(image_dict)>0:
            print ("SHOW SUCESS")
            await cl.Message(content=answer).send()
            run_sync(show_image(answer , image_dict))
            cl.user_session.set("chain", chain)
        else:
            await cl.Message(content=answer).send()
            cl.user_session.set("chain", chain)
