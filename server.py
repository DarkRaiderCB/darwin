import os
import json
from uuid import uuid4
from typing import List
import ast
from together import Together
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fastapi
from decouple import config
from gridfs import GridFS
from PIL import Image
import io
import mimetypes
import pandas as pd
import asyncio
from utils.process import *
from utils.fileparse import *
from utils.parse_function import extract_function_names, extract_function_parameters, extract_iter
from functions.coder import *
from functions.web_api import *
from functions.call_function import function_dict
from functions.extract_web_links import extract_links, scrape_pdf
import copy
import traceback


# Initialize FastAPI app
app = FastAPI()
global history
global web_search_response
global StateOfMind
web_search_response = ""
history = ""
StateOfMind = ""
cc = 0
iter = 0
prevcoder = False

# Enable CORS for all routes
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pickledb
db = pickledb.load('./data/data.db', True) 

def update_db(project_name, val):
    project = db.get(project_name)
    project.append(val)
    db.dump()
    
def get_db(project_name):
    project = db.get(project_name)
    return copy.deepcopy(project)

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv("MODEL")
MAX_TOKENS = 10000
TEMPERATURE = 0

def chatGPT(project_name, original_query):
    global history
    global web_search_response
    global StateOfMind
    global cc
    global iter
    prevcall = None
    client = Together()  # Initialize Together API client

    while True:
        iter += 1
        prompt = process_assistant_data(original_query, StateOfMind, prevcall)
        history = get_db(project_name)
        history_string = ""
        for obj in history:
            history_string += f"User: {obj['user_query']}\n" if "user_query" in obj else ""
            history_string += f"AI_Coder_Message: {obj['message']}\n" if "message" in obj else ""
            history_string += f"AI_Coder_Code: {obj['code']}\n" if "code" in obj else ""
            history_string += f"AI_Coder_Output: {obj['console']}\n" if "console" in obj else ""
            history_string += f"Web_search: {obj['web_search']}\n" if "web_search" in obj else ""

        print("history", history_string)
        message = [
            {"role": "system", "content": history_string},
            {"role": "user", "content": prompt}
        ]
        print("\nMessage to Together: \n", prompt)

        # Call Together API
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=message,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|im_end|>"],
            stream=True
        )

        result = ""
        for token in response:
            if hasattr(token, "choices"):
                delta_content = token.choices[0].delta.content
                print(delta_content, end="", flush=True)
                result += delta_content

        print("\n\nResult: \n", result)
        functions = extract_function_names(result)
        print("\nFunctions: \n", functions)

        if functions:
            parameters = extract_function_parameters(result)
            func = functions[0]
            parameter = parameters[0]
            print(func)
            print(parameter)
            try:
                if func == "coder":
                    prevcall = "coder"
                    query = parameter['query']
                    coder = Coder(project_name)
                    for chunk in coder.code(query, web_search_response):
                        if json.loads(chunk) == {"exit": True}:
                            break
                        yield chunk
                        update_db(project_name, json.loads(chunk))
                    StateOfMind = "The coder function took the following steps :\n" + coder.summary
                    prevcoder = True
                    cc += 1
                    if cc >= 2:
                        StateOfMind = "Coder call finished. Call the summary_text function!"
                    
                elif func == "web_search":
                    response = web_search(parameter['query'])
                    out = json.dumps({"web_search": str(response)})
                    yield out.encode("utf-8") + b"\n"
                    update_db(project_name, {"web_search": str(response)})
                    web_search_response = response
                    StateOfMind = "Browsed the web and retrieved relevant information. Call the coder function or return to user."
                    prevcall = "web_search"
                    
                elif func == "summary_text":
                    prevcall = "summary_text"
                    response = parameter['message']
                    response = response.replace("", "")
                    out = json.dumps({"summary_text": response})
                    yield out.encode("utf-8") + b"\n"
                    update_db(project_name, {"summary_text": response})
                    yield b''
                    cc = 0
                    iter = 0
                    break

                elif func == "getIssueSummary":
                    from functions.issues import issueHelper
                    prevcoder = False
                    statement = parameter['statement']
                    issue_helper = issueHelper(project_name)
                    issue_summary = issue_helper.getIssueSummary(statement)
                    StateOfMind = "The issue details have been extracted as follows : \n" + issue_summary + "\n\n NOTE : Call the summary_text function to answer user's query or Coder function to solve the issue."
                    out = json.dumps({"getIssueSummary": issue_summary})
                    yield out.encode("utf-8") + b"\n"
                    update_db(project_name, {"getIssueSummary": issue_summary})
            except Exception as e:
                print(f"Error calling the function {func} with parameters {parameter}: {e}")
                traceback.print_exc()
        else:
            pass


# Start the server
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    uvicorn.run(app, host="0.0.0.0", port=parser.parse_args().port)
