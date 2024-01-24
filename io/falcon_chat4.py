
################################################################################
# Author: Darrell O. Ricke, Ph.D.  (email: Darrell.Ricke@ll.mit.edu)
#
# RAMS request ID  1026639
# 
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# This material is based upon work supported by the Department of the Air Force 
# under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, 
# conclusions or recommendations expressed in this material are those of the 
# author(s) and do not necessarily reflect the views of the Department of the 
# Air Force.
# 
# © 2024 Massachusetts Institute of Technology.
# 
# The software/firmware is provided to you on an As-ls basis
# Delivered to the U.S. Government with Unlimited Rights, as defined in 
# DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright 
# notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 
# or DFARS 252.227-7014 as detailed above. Use of this work other than as 
# specifically authorized by the U.S. Government may violate any copyrights that 
# exist in this work.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
################################################################################

import json
import transformers
import torch
import os
import sys
from os import environ
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.memory.chat_message_histories.file import FileChatMessageHistory

from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

from InputFile import InputFile

# model_id = "tiiuae/falcon-180B"
model_id = "tiiuae/falcon-7b"

# environ["TRANSFORMERS_OFFLINE"] = "1"
# environ["TRANSFORMERS_CACHE"] = "/raid/LLM/Models/falcon/projects"
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2" # if you need to specify GPUs

################################################################################
def clean( message ):
  i = message.find( "[HumanMessage" )
  if i > 0:
    message = message[ 0:i ]
  i = message.find( " The above example is a chatbot." )
  if i > 0:
    message = message[ 0:i ]
  return message.replace( "\n", " " )

################################################################################
# This function reads in a JSON file.
def read_json( filename ):
  f = open( filename )
  data = json.load( f )
  f.close()
  return data

################################################################################
# This function reads in a text file.
def read_text( filename, as_string ):
  df = InputFile()
  df.setFileName( filename )
  df.openFile()
  if as_string:
    df.readText()
    df.closeFile()
    return df.contents
  else:
    df.readArray()
    df.closeFile()
    return df.lines

################################################################################
def parse_memory( memory ):
  human_message = ""
  for message in memory:
    print( f"message: {message}, type: {type(message)}" )

################################################################################
arg_count = len(sys.argv)
if ( arg_count >= 4 ):
  token_text = read_text( sys.argv[1], as_string=True )
  template = read_text( sys.argv[2], as_string=True )
  print( "Template:" )
  print( template )

  tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_text, load_in_16bit=True, trust_remote_code=True, device_map="auto", )

  pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    # max_length=1000,
    max_new_tokens=200,
    do_sample=True,
    top_k=4,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
  )

  hf_llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

  prompt = PromptTemplate(template=template, input_variables=["memory","question"])
  cbm_memory = ConversationBufferMemory(memory_key="memory", input_key="question", return_messages=True, ai_prefix='AI', human_prefix="Human" )

  # Load in past chat questions and answers.
  if ( arg_count >= 5 ):
    history_file = sys.argv[4]
    if os.path.isfile( history_file ):
      chat_mem = read_text( sys.argv[4], as_string=False )
      print( "***** Messages:" )
      for message in chat_mem:
        parts = message.split( "\t" )
        if len( parts ) == 2:
          print( message )
          cbm_memory.chat_memory.add_user_message( parts[0] )
          cbm_memory.chat_memory.add_ai_message( parts[1] )

  llm_chain = LLMChain(prompt=prompt, llm=hf_llm, memory=cbm_memory)

  questions = read_text( sys.argv[3], as_string=False )
  for question in questions:
  #   print( "-----------------------------------------------------" )
  #   print( "Question: " + question )
  #   print(llm_chain.run(question))
    llm_chain.run(question)

  # print( "-----------------------------------------------------" )
  # print( "Chat history messages buffer:" )
  # print( llm_chain.memory.buffer )
  # print( "-----------------------------------------------------" )

  
  mem_var = cbm_memory.load_memory_variables( {} )
  # print( f"memory variables: {mem_var}" )
  parse_memory( mem_var['memory'] )
  # print( "-----------------------------------------------------" )

  if ( arg_count >= 5 ):
    with open( sys.argv[4], "w" ) as f:
      human_message = ""
      for message in mem_var['memory']:
        if type( message ) == AIMessage:
          ai_message = clean( message.content )
          # print( f"ai_message: |{ai_message}|" )
          f.write( human_message + "\t" + ai_message + "\n" )
          human_message = ""
        else:
          human_message = message.content
          # print( f"human_message: |{human_message}|" )
else:
  print( "usage: python falcon_chat4.py <token file> <template file> <questions file> <memory file>" )
