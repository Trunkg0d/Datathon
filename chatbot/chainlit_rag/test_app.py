# import chainlit as cl

# @cl.action_callback("action_button")
# async def on_action(action):
#     await cl.Message(content=f"Executed {action.name}").send()
#     await action.remove()
#     await cl.Message(content=f"Đã mở phòng thử đồ {action.name}").send()

# @cl.action_callback("inter_virtual_fittingroom")
# async def on_action(action):
#     await cl.Message(content=f"Executed {action.name}").send()
#     await action.remove()

# @cl.on_chat_start
# async def start():
#     actions = [
#         cl.Action(name="action_button", value="example_value", description="Click me!"),
#         cl.Action(name="inter_virtual_fittingroom", value="example_value", description="Click me!")
#     ]

#     await cl.Message(content="Interact with this action button:", actions=actions).send()


# from langchain.tools import DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()
# print ( search.run("Outfut rating criteria") )

from sentence_transformers import SentenceTransformer , util
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

user_input = ["I want a comfortable T-shirt, which is formal but also easy to wear"]

sentence_2_compare = ['Help user go to Virtual fitting room, try on' , 
                      'Provide the user input query, please recommend outfit',
                      'Provide the user input query image, recommend outfit for user']

user_embedding = model.encode (user_input)
scores = []
for sentence in sentence_2_compare :
    product_embedding = model.encode (sentence_2_compare)
    scores.append ( util.pytorch_cos_sim(user_embedding, sentence )[0][0].item() )
    
print (scores)
print (np.argmax (np.array (scores)) )