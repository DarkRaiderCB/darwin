import os
import json
from uuid import uuid4
from typing import List
import ast
from openai import OpenAI
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
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
from functions.issues import issueHelper
import copy


# Initialize FastAPI app
app = FastAPI()
global history
global web_search_response
global StateOfMind
web_search_response = ""
history = ""
StateOfMind = ""

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

# Constants
MODEL_NAME =  "gpt-4-turbo"  # config('MODEL_NAME')
MAX_TOKENS = 10000
TEMPERATURE = 0

def convert_bytes_to_original_format(file_bytes, mime_type, save_path):
    if mime_type.startswith('text'):
        # Save text file
        with open(save_path, 'w', encoding='utf-8') as text_file:
            text_file.write(file_bytes.decode('utf-8'))
    elif mime_type.startswith('image'):
        # Save image file
        img = Image.open(io.BytesIO(file_bytes))
        img.save(save_path)
    elif mime_type == 'application/json':
        # Save JSON file
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(json.loads(file_bytes.decode('utf-8')), json_file, indent=2)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        # Save Excel file
        df = pd.read_excel(io.BytesIO(file_bytes))
        df.to_excel(save_path, index=False)
    elif mime_type == 'application/pdf':
        # Save PDF file
        # Example: pdf_to_text_and_save(file_bytes, save_path)
        pass
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Save DOCX file
        doc = Document(io.BytesIO(file_bytes))
        doc.save(save_path)
    else:
        # Handle other types or raise an exception for unknown types
        raise ValueError(f"Unsupported MIME type: {mime_type}")



def store_file_in_mongodb(file_path, collection_name):
    fs = GridFS(db, collection=collection_name)
    with open(file_path, 'rb') as file:
        # Determine file type using mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        metadata = {'mime_type': mime_type}

        file_id = fs.put(file, filename=os.path.basename(file_path), metadata=metadata)

    return file_id

def retrieve_file_from_mongodb(file_id, collection_name):
    fs = GridFS(db, collection=collection_name)
    file_data = fs.get(file_id)

    # Retrieve metadata
    metadata = file_data.metadata
    mime_type = metadata.get('mime_type', 'application/octet-stream')

    return file_data.read(), mime_type


def get_folder_structure(dir_path,parent=""):
    is_directory = os.path.isdir(dir_path)
    name = os.path.basename(dir_path)
    relative_path = os.path.relpath(dir_path, os.path.dirname(dir_path))

    directory_object = {
        'parent': os.path.dirname(dir_path),
        'path': relative_path,
        'name': name,
        'type': 'directory' if is_directory else 'file',
    }

    if is_directory:
        children = []
        for child in os.listdir(dir_path):
            child_path = os.path.join(dir_path, child)
            children.append(get_folder_structure(child_path,parent=dir_path))
        directory_object['children'] = children

    return directory_object


@app.post("/serve_file")
async def serve_file(request: Request):
    data = await request.form()
    filePath = data["path"]
    # serve file using fastapi FileResponse
    pwd = os.path.join(os.getcwd(), 'data')
    path = os.path.join(pwd, filePath)
    return fastapi.responses.FileResponse(path)

@app.post("/folder_structure")
async def folder_structure(request:Request):
    data = await request.form()
    root_dir = data["root_dir"]
    structure = get_folder_structure(root_dir)
    return structure

@app.get("/get_file")
async def get_file(request: Request):
    data = await request.form()
    path = data["path"]
    with open(path, "rb") as file:
        file_bytes = file.read()
    return file_bytes

@app.post("/create_project") # creates a new project and updates the global state with the project data
async def create_project(request: Request):
    data = await request.form()
    project_name = data.get("project_name")
    # check if project already exists
    for key in db.getall():
        if key == project_name:
            return {"message": "Project already exists"}
    db.set(project_name,[])
    return {"project_name": project_name}

@app.post("/get_project_data") # updates the global state with the project data
async def get_project(request: Request):
    data = await request.form()
    project_name = data.get("project_name")
    project = db.get(project_name)
    print(project)
    return project

@app.delete("/delete_project") # deletes the project
async def delete_project(request: Request):
    data = await request.form()
    project_name = data.get("project_name")
    for key in db.getall():
        if key == project_name:
            db.rem(key)
            return {"message": "Project deleted successfully"}
    return {"message": "Project not found"}
    
@app.get("/get_project_names") # returns key value pairs of id and project name
async def get_projects():
    list = []
    for key in db.getall():
        project_name = key
        list.append(project_name) 
    return list

@app.post("/chat")

async def chat(request: Request,file: UploadFile = None,image: UploadFile = None):
    """
    FORM DATA FORMAT:
    {
        "project_name": "project_name",
        "customer_message": "message"
    }
    """
    data = await request.form()
    project_name = data.get("project_name")
    customer_message = data.get("customer_message")
    server_response = ""
    function_response = ""
    coder_response = list()
    web_search_response = ""
    global StateOfMind 
    if(StateOfMind == ""):
        StateOfMind = customer_message
    original_query = customer_message
    global history
    history = get_db(project_name)
    while(True):
        res = await chatGPT(project_name,original_query)
        function_response = res.get("function_response")
        functions = res.get("functions")
        # chat = add_chat_log(func, function_response.get(func), chat)
        # server_response = function_response.get(functions)
        if('coder' in functions):
            coder_response.append(function_response["coder"])
            StateOfMind = "Coder Response : " + res["StateOfMind"]
            print("StateOfMind", StateOfMind)
            print("coder_response ok")
        if('web_search' in functions):
            web_search_response = function_response["web_search"]
            StateOfMind = res["StateOfMind"]
            print("web_search_response ok")
        if('getIssueSummary' in functions):
            StateOfMind = "I have extracted the Issue details as follows : " + function_response["getIssueSummary"] + "\n"
            print("getIssueSummary ok")
        if('summary_text' in functions):
            server_response = {"user_query":customer_message,"summary":function_response["summary_text"], "coder_response":coder_response, "web_search_response":web_search_response}
            update_db(project_name,server_response)
            history.append(server_response)
            return server_response

def add_chat_log(agent, response, chat_log=""):
    return f"{chat_log}{agent}: {response}\n"

async def chatGPT(project_name,original_query):
    global history
    global web_search_response
    global StateOfMind
    history_string = ""
    for obj in history:
        history_string += f"User: {obj['user_query']}\n" if obj['user_query'] else ""
        history_string += f"AI: {obj['summary']}\n" if obj['summary'] else ""
        history_string += f"AI: {obj['coder_response']}\n" if obj['coder_response'] else "" 
        history_string += f"AI: {obj['web_search_response']}\n" if obj['web_search_response'] else ""
    res ={
        "result":None,
        "function_response":None,
        "functions":None,
        "StateOfMind":None
    }
    prompt = process_assistant_data(original_query,StateOfMind)
    print("history" ,history_string)
    message = [
        {"role": "system", "content": history_string},
        {"role": "user", "content": prompt}
    ]
    print("\nMessage to GPT: \n", prompt)
    gpt_response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=message,
        temperature=TEMPERATURE,
    )
    print("\ngpt_response",gpt_response)
    result = gpt_response.choices[0].message.content.strip()
    print("\n\nResult: \n", result)

    functions = extract_function_names(result)
    res["result"] = result    

    # print("\nChat: \n", chat)    
    print("\nFunctions: \n", functions)
    res["functions"] = functions
    function_response = dict()
    
    if functions:
        parameters = extract_function_parameters(result)
        for func, parameter in zip(functions, parameters):
            print(func)
            print(parameter)
            try:
                if func == "coder":
                    query = parameter['query']
                    coder = Coder(project_name)
                    coder_response = coder.code(query,web_search_response)
                    parsed = coder.parse_output(coder_response)
                    function_response.update({"coder":parsed})
                    res["StateOfMind"] = coder.generate_summary(parsed)
                    # function_response.update({"summary_text":coder.generate_summary(parsed)})
                elif func == "web_search":
                    response = (web_search(parameter['query']))
                    web_search_response = response
                    function_response.update({"web_search":response})
                    res["StateOfMind"] = "Browsed the web and retrieved relevant information. Call the coder."
                elif func == "summary_text":
                    response = (parameter['message'])
                    function_response.update({"summary_text":response})
                elif func == "getIssueSummary":
                    statement = parameter['statement']
                    issue_helper = issueHelper(project_name)
                    response = issue_helper.getIssueSummary(statement)
                    function_response.update({"getIssueSummary":response})
                else:
                    pass
            except Exception as e:
                print(f"Error calling the function {func} with parameters {parameter}: {e}")
    res["function_response"] = function_response
    return res

# start the server
if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    print(os.environ["OPENAI_API_KEY"])
    openai = OpenAI(api_key=openai_api_key)
    uvicorn.run(app, host="0.0.0.0", port=8080)
