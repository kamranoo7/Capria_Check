import base64
import requests
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from flask import request, jsonify
from google.oauth2.credentials import Credentials
from flask import Flask, request, send_file
from google.api_core.exceptions import BadRequest
import requests
import PyPDF2
from io import BytesIO
import pandas as pd
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text_to_fp
from pdfminer.high_level import extract_text
import glob
import docx2txt
import iso8601
import shutil
import zipfile
import logging
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from flask import Flask, request, render_template, send_file
from datetime import datetime
import time
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    Response,
)
from openpyxl.styles import Font
import os
from datetime import datetime, timezone
import weaviate
import openai
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import io
import re
import threading
import random
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import secrets
import uuid
from dotenv import load_dotenv
import ast
import fitz  # PyMuPDF
import os
import shutil
import json
from auth import login_required, admin_required, user_activity_tracker
from admin import (
    updateAdminFlag,
    getAdminFlag,
    updateAdminLastActivity,
    getAdminLastActivity,
)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timezone

import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError


nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)

# log = logging.getLogger("werkzeug")
# log.setLevel(logging.ERROR)


mail = Mail(app)
app.secret_key = "JobBot"

global lm_client
global layer_1
global loading_status


loading_status = None


import tracemalloc

tracemalloc.start()


load_dotenv()


app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587  # Use 465 for SSL
app.config["MAIL_USERNAME"] = "developers@capria.vc"  # Ensure correct email
app.config["MAIL_PASSWORD"] = "ladn pjid owpg kuos"
app.config["MAIL_USE_TLS"] = True  # Set to False if using SSL
app.config["MAIL_USE_SSL"] = False  # Set to True if using SSL


mail.init_app(app)

global audio_speech
audio_speech = None

global projectName
projectName = "Bot"

global p1
p1 = "provide the answer only in the context of Capria Global  South Fund II"

global p2
p2 = "provide the answer only in the context of Capria Global  South Fund II"

global l1_text, l2_text, level2_search_prompt
l1_text = "Level 1: "
l2_text = "Level def: "
level2_search_prompt = "Search within level 2?"

global stop_flag, check_for_whitelist
stop_flag = False
check_for_whitelist = False

global citations_dictt

import os


# @app.before_request
# def before_request_log():
#     print(f"\n\n\n\nRoute {request.path} is being invoked\n\n\n\n\n")


def readAndWriteJsonData(path, mode, data=None):
    if mode == "r":
        try:
            with open(path, "r") as file:
                json_data = json.load(file)
            return json_data
        except FileNotFoundError:
            print(f"File '{path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from '{path}'.")
            return None
    elif mode == "w":
        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Data written to '{path}' successfully.")
        except Exception as e:
            print(f"Error writing data to '{path}': {e}")
    else:
        print("Invalid mode. Use 'r' for reading or 'w' for writing.")


def manage_json(action, field_name, value=None, file_path="progress_log.json"):
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print("Error reading JSON file. File could be empty or corrupt.")
            return "Error reading JSON file."

    if action == "write":
        data[field_name] = value
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        return None  # Optionally return a success message or confirmation
    elif action == "read":
        try:
            return data[field_name]
        except KeyError:
            return "Field not found."
    else:
        return "Invalid action specified."


def extract_digits(text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, text)
    digits = []
    for match in matches:
        digits.extend([int(digit) for digit in match.split(",")])
    return list(set(digits))


def delete_pdfs_and_docxs(folder_path):
    # Check if the folder exists
    try:
        if not os.path.isdir(folder_path):
            print("The specified folder does not exist.")
            return

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a PDF or DOCX
            if filename.endswith(".pdf") or filename.endswith(".docx"):
                # Construct full file path
                file_path = os.path.join(folder_path, filename)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted {file_path}")
    except Exception as e:
        print("\n\nDeletion error:: {} \n\n".format(e))


def cleanDir():
    email = manage_json("read", "instanceRunning")
    delete_pdfs_and_docxs(f"Layer_1_file_{email}")
    delete_pdfs_and_docxs(f"Layer_2_file_{email}")


cleanDir()


def progress_log(new_text, file_path="loading_status.txt", mode="write"):
    if mode == "read":
        try:
            with open(file_path, "r") as file:
                original_content = file.read()
        except:
            original_content = "..."
        return original_content

    with open(file_path, "w") as file:
        file.write(new_text)
    return None


progress_log("Ready to use...")


manage_json("write", "adminLoginCount", value=0)

custom_functionsz = [
    {
        "name": "return_response",
        "description": "to be used to return list of words/tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of tags directly extracted from the doctextument given",
                    "items": {"type": "string"},
                },
            },
            "required": ["item_list"],
        },
    }
]


def word_client():
    existing_data = readAndWriteJsonData("configu.json", "r")

    layer_1 = existing_data["layer1"]

    openaiKey = existing_data["openaiKey"]

    client = weaviate.Client(
        url=layer_1["layer1URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=layer_1["layer1AuthKey"]),
        additional_headers={"X-OpenAI-Api-Key": openaiKey},
    )
    return client


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def find_keys_containing_word(word_dict, word, metadata_dictt):
    word_lower = word.lower()
    keys_with_word = []
    metadata = []
    for key, words in word_dict.items():
        if word_lower in [w.lower() for w in words]:
            if key not in keys_with_word:
                keys_with_word.append(key)
                met = metadata_dictt[key]
                metadata.append(met)
    return keys_with_word, metadata


def ask_gpt_tags(question):
    system_message = "You will be given a sentence. You need to return the list of tags from it. This will be used for tagging/indexing this peice of text. When someone searches for information, these tags will be used to filter the text. Return a python list. These tags should be names. locations, position, or the subject. Do not add adjectives or descriptors. This will be a high level indexing. As as example: 'Applied GenAI' or 'GenAI Problems' will be tagged as  just 'GenAI'. Maybe I will create subgroups later. But, for now its only high level groupings."
    user_message = "content below: \n" + question
    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    print("-----------------------")
    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        max_tokens=500,
        temperature=0.0,
        functions=custom_functionsz,
        function_call="auto",
    )

    reply = response.choices[0].message.content
    try:
        reply = ast.literal_eval(reply)
        print("Regular text to list.")
        try:
            reply = reply["item_list"]
        except:
            pass
        print(reply)
    except:
        try:
            reply = json.loads(response.choices[0].message.function_call.arguments)[
                "item_list"
            ]
            print(reply)
        except Exception as e:
            print(e)
            reply = []
    return reply


def qdb_tags(query, db_client, name, cname, limit):
    data = []
    try:
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )

        print(res, "response form qdb_tags")

        data = []
        
        for i in range(len(res["data"]["Get"][cname])):
            met = res["data"]["Get"][cname][i]["text"]
            data.append(met)

        print(data,'data form qdb_tags---------')
    except Exception as e:
        print("Exception in DB, dude.")
        print(e)
    return data


def get_relevant_tags(tags, class_name, class_name_capitalized):
    alltags = []
    for tag in tags:
        alltags.extend(
            qdb_tags(tag, word_client(), class_name, class_name_capitalized, 4)
        )
    return alltags


def get_all_sentences(words, data):
    print("----------++-----------")
    # email = session.get(
    #     "current_email", session.get("email")
    # )  
    email = manage_json("read", "instanceRunning")
    print(f"Using email for operations: {email}")  # Debug prin
    print(email, "emailfor sentencce.json ")
    metadata_dictt = read_json_file(f"flipped_dict_{email}.json")
    all_sentences = []
    all_metadata = []
    for word in words:
        sentence, metadata = find_keys_containing_word(
            data, word.lower(), metadata_dictt
        )
        all_sentences.extend(sentence)
        all_metadata.extend(metadata)
    return remove_duplicates_preserve_order(
        all_sentences
    ), remove_duplicates_preserve_order(all_metadata)


def join_strings_with_ids(string_list):
    formatted_strings = [
        f"chunk id: {index} \n {value}" for index, value in enumerate(string_list)
    ]
    joined_string = "\n\n".join(formatted_strings)
    return joined_string


def progress_log(new_text, file_path="loading_status.txt", mode="write"):
    if mode == "read":
        try:
            with open(file_path, "r") as file:
                original_content = file.read()
        except:
            original_content = "..."
        return original_content

    with open(file_path, "w") as file:
        file.write(new_text)
    return None


def generate_google_drive_url(file_id):
    return f"http://drive.google.com/file/d/{file_id}/view"


@app.route("/send_verification", methods=["POST"])
def send_verification():
    try:
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"message": "No email provided"}), 400

        print(email, "------------asdfasdfas")

        if check_email_exists(email):
            print("Name already exists")
            message = "Email already exists."
            return jsonify({"message": message}), 400

        otp = random.randint(100000, 999999)
        session[email] = otp
        print(otp, "otp sent ")

        msg = Message(
            "Email Verification", sender="developers@capria.vc", recipients=[email]
        )
        msg.body = f"Your verification code is: {otp}"
        mail.send(msg)

        return jsonify({"message": "Verification email sent"})
    except Exception as e:
        print(e, "erron while /send_verification")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 400


@app.route("/send_verification1", methods=["POST"])
def send_verification1():
    try:
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"message": "No email provided"}), 400

        if not check_email_exists(email):
            return jsonify({"message": "Email does not exist."}), 400

        otp = random.randint(100000, 999999)
        session[email] = otp
        print(f"Generated OTP for {email}: {otp}")
        msg = Message(
            "Email Verification", sender="developers@capria.vc", recipients=[email]
        )
        msg.body = f"Your verification code is: {otp}"
        mail.send(msg)

        return jsonify({"message": "Verification email sent"})
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 400


def replace_in_string(text):
    chars_to_replace = ["\\", "/", "_"]
    for char in chars_to_replace:
        text = text.replace(char, " ")
    text = text.replace(".png", "")
    return text


@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    print(otp, "otp ")
    stored_otp = session.get(email)
    print(stored_otp, "stored_otp")

    if stored_otp and str(stored_otp) == str(otp):
        return jsonify({"message": "Email verified successfully"})
    else:
        return jsonify({"message": "Invalid or expired OTP"}), 400


key = os.getenv("SECRET_KEY")

# lm_client = openai.OpenAI(api_key=key)
service_account_file = "cred.json"
credentials = service_account.Credentials.from_service_account_file(
    service_account_file
)
dbclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
credentials_path = "credentials.json"
response = "Done."
app.secret_key = "secret_key"

global intro
intro = """
Welcome to the alpha version of Capria's DD Copilot. Trying asking "What is the MOFC for Betterplace" or "Does Capria invest in genai infrastructure?" or "Does Capria consider DEI" or "Who is Capria's tax advisor"
"""


def add_row_to_sheet(data, sheet_id, sheet_name="Sheet1"):
    """Appends a row of data to a specific Google Sheet."""
    try:
        # Define the scope and create credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        client = gspread.authorize(creds)

        # Open the spreadsheet and the specific worksheet
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.worksheet(sheet_name)  # Using worksheet by name for clarity

        # Append the data as a new row in the worksheet
        worksheet.append_row(data)
        print("Row added successfully!")
    except gspread.exceptions.APIError as e:
        print("API Error:", e)
    except gspread.exceptions.SpreadsheetNotFound:
        print("Spreadsheet not found. Check your sheet ID.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found. Check your sheet name.")
    except Exception as e:
        print("An error occurred:", e)


from google.cloud import bigquery


def get_user_by_userID(userID):
    try:

        dataset_name = "jobbot-415816.jobBot"
        table_name = "users"

        query = """
            SELECT *
            FROM `{0}.{1}`
            WHERE userID = @userID
        """.format(
            dataset_name, table_name
        )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("userID", "STRING", userID)]
        )

        query_job = dbclient.query(query, job_config=job_config)  # Make an API request

        results = query_job.result()  # Waits for job to complete

        for row in results:
            return {
                "password": row.password,
                "userID": row.userID,
                "isAnswer": row.isAnswer,
                "role": row.role,
            }

        return None

    except Exception as e:
        print(f"An error occurred while retrieving user: {str(e)}")
        return None


@app.route("/control_panel", methods=["GET", "POST"])
@login_required
@user_activity_tracker
def control_panel():
    global intro, projectName, p1, p2, l1_text, l2_text, level2_search_prompt

    # Load configuration for GET request
    config_data = {"layer1": {}, "layer2": {}}

    # Load configuration
    config_file_path = "configu.json"
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

    if request.method == "POST":
        print(request.form)

        l1_text = request.form.get("text_level1", "Level 1:")
        l2_text = request.form.get("text_level2", "Level 2:")
        session["language"] = request.form.get("language", "english")
        session["language"] = get_language(session["language"])
        projectName = request.form.get("title", "Bot")
        session["intro"] = request.form.get("intro", "")
        intro = session["intro"] if session["intro"] != "" else intro
        level2_search_prompt = request.form.get(
            "level2_search_prompt", "Search within Level 2?"
        )

        p1 = request.form.get("prompt_level1", "")
        p2 = request.form.get("prompt_level2", "")
        manage_json("write", "l1", value=l1_text)
        manage_json("write", "intro", value=intro)
        manage_json("write", "l2", value=l2_text)
        manage_json("write", "p1", value=p1)
        manage_json("write", "p2", value=p2)
        manage_json("write", "intro", value=intro)
        manage_json("write", "name", value=projectName)
        manage_json("write", "searchprompt", value=level2_search_prompt)

        existing_data = readAndWriteJsonData("configu.json", "r")

        progress_log_data = read_json_file("progress_log.json")

        email = session.get("email","")

        user_config = {
        "username": email,
        "language": get_language(session.get("language", "en")),
        "projectName": progress_log_data["name"],
        "intro": progress_log_data["intro"],
        "level2_search_prompt": progress_log_data["searchprompt"],
        "l1_text": progress_log_data["l1"],
        "l2_text": progress_log_data["l2"],
        "p1": progress_log_data["p1"],
        "p2": progress_log_data["p2"],
        "layer1_config": existing_data["layer1"],
        "layer2_config": existing_data["layer2"],
        "vectorFileChange": progress_log_data['vectorFileChange'],
        "openaiKey": existing_data['openaiKey'],
        "email":email
    }

        save_user_config(email, user_config)


    instanceRunning = manage_json("read", "instanceRunning")
    instanceUsers = get_usernames()
    email = session["email"]

    print("\n\nLanguage being retuned:", session["language"], "\n\n")
    userID = session["userID"]
    user = get_user_by_userID(userID)
    role = user["role"]
    session["level_1_text"] = (
        session["level_1_text"] if "level_1_text" in session else "Level1:"
    )
    session["level_2_text"] = (
        session["level_2_text"] if "level_2_text" in session else "Level2:"
    )
    print("\n\n", l1_text, l2_text, "\n\n")
    return render_template(
        "control_panel.html",
        language=session.get("language", "english"),
        intro=manage_json("read", "intro"),
        prompt_level1=manage_json("read", "p1"),
        prompt_level2=manage_json("read", "p2"),
        prompt_level3=manage_json("read", "p1"),
        role=role,
        name=manage_json("read", "name"),
        config_data=config_data["layer1"],  # Pass layer1 config
        config_data1=config_data["layer2"],
        flag=manage_json("read", "whitelist"),
        layer=manage_json("read", "layerLevel"),
        text_level1=manage_json("read", "l1"),
        text_level2=manage_json("read", "l2"),
        level2_search_prompt=manage_json("read", "searchprompt"),
        instanceRunning=instanceRunning if instanceRunning else "",
        users=instanceUsers,
        openAIkey=config_data["openaiKey"] if "openaiKey" in config_data else "",
        userLogin=session.get("email", ""),
    )


@app.route("/trans")
def trans():
    global intro, projectName, level2_search_prompt
    newList = []
    transwords = [
        manage_json("read", "name"),
        "User",
        "Enter Your Query",
        "Feedback",
        "Fast Mode",
        manage_json("read", "intro"),
        manage_json("read", "searchprompt"),
        "Submit",
        "Close",
    ]

    if session["language"] == "null":
        session["language"] = "en"

    if session["language"] != "en":
        for item in transwords:
            trans = translate_text(item, session["language"])
            print(trans, "transscripted words ")
            newList.append(trans)

    if not newList:
        newList = transwords

    return jsonify(newList)


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    try:
        audio_file = request.files["audioFile"]
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = FileWithNames(audio_bytes)
            session["greet"], session["language"] = transcribe(audio_file)
            session["language"] = (
                request.form["language"]
                if request.form["language"] != "auto"
                else session["language"]
            )
            session["language"] = get_language(session["language"])
            print("\n\n\n", session["language"], "\n\n\n")

            print("\n\n\n", session["language"], "\n\n\n")
    except Exception as e:
        print(e)
        update_logs(e)
        session["language"] = "english"

    return jsonify({"channel": "chat"})


@app.route("/chat")
@login_required
@user_activity_tracker
def chat():
    global layerLevel
    if "language" not in session or session["language"] is None:
        session["language"] = "en"
        print("Language set to English")
    userID = session["userID"]
    user = get_user_by_userID(userID)
    # print("users---------", user)
    role = user["role"]
    print("Redirecting...")
    return render_template(
        "chat.html", role=role, layer=manage_json("read", "layerLevel")
    )


def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@app.route("/")
def index():
    return redirect("login")


custom_functions = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of chunk ids. ONLY the ones used to generate the response to the question being asked. return the id only if the info was used in the response. think carefully.",
                    "items": {"type": "integer"},
                },
                "response": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient", "item_list"],
        },
    }
]

custom_functions_1 = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "boolean",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient"],
        },
    }
]

import time

custom_functionsz = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response_answer": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "item_list": {
                    "type": "array",
                    "description": "List of chunk ids. ONLY the ones used to generate the response to the question being asked. return the id only if the info was used in the response. think carefully.",
                    "items": {"type": "integer"},
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response_answer", "sufficient", "item_list"],
        },
    }
]


def extract_digits(text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, text)
    digits = []
    for match in matches:
        digits.extend([int(digit) for digit in match.split(",")])
    return list(set(digits))


def ask_gpt(
    question,
    context,
    gpt,
    metadata,
    language,
    addition,
    userid,
    filename="Layer_1_file/file_info_1.json",
    intro_msg="",
):

    if filename == "Layer_1_file/file_info_1.json":
        email = manage_json("read", "instanceRunning")
        filename = f"Layer_1_file_{email}/file_info_1.json"

    global audio_speech, stop_flag, citations_dictt
    user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
    system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. provide in-text citations within square brackets at the end of each sentence, right after each fullstop. The citation number represents the chunk id that was used to generate that sentence. Do Not bunch multiple citations in one bracket. Uee seperate brackets for each digit. {} Return the response along with a boolean value indicating if the information from the context was enough to answer the question. Return true if it was, False if it wasnt. Return the response, which is th answer to the question asked".format(
        addition
    )

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    print("\n\n\n\n\n\nChecking for response\n\n\n\n\n\n")

    response = lm_client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=msg,
        max_tokens=2000,
        temperature=0.0,
        seed=1,
        functions=custom_functionsz,
        function_call="auto",
    )

    reply = ""
    reply += "\n\n"
    reply += intro_msg
    reply += "\n\n"
    print(
        "=========================\n\n\n\n\n\n",
        reply,
        "\n\n\n\n\n+++++==========================",
    )
    try:
        resp_json = json.loads(response.choices[0].message.function_call.arguments)
        item_list = resp_json["item_list"]
        reply += resp_json["response_answer"]
    except Exception as e:
        print(e)
        item_list = []
        reply += response.choices[0].message.content
        reply = reply.split("item_list")[0]
        try:
            item_list = extract_digits(reply)
            print(item_list)
        except Exception as e:
            print("-----------eeeee------------", e)
            item_list = []

    with open(filename, "r") as file:
        files_metadata = json.load(file)

    try:
        print(item_list)
        cits = ["www.google.com"] * (max(item_list) + 4)
        for item in item_list:
            reply += "\n"
            reply += "[{}]".format(item)
            print(metadata[item])
            reply += replace_in_string(metadata[item].split(".png")[0])
            for filedata in files_metadata:
                name = filedata["name"].split(".docx")[0]
                name = filedata["name"].split(".pptx")[0]
                name = filedata["name"].split(".pdf")[0]
                file_id = filedata["id"]
                if name in metadata[item]:
                    cits[item] = generate_google_drive_url(file_id)

        lst = "\n\n\n list_of_citations = " + str(cits)
        lst = lst.replace("'", '"')
        reply += lst
    except:
        pass

    # print(response)
    data = {"response": reply, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"


def qdb(query, db_client, name, cname, chunk_id, limit, retry_count=3):
    context = None
    metadata = []
    try:
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )
        print(res, "response form qdb")

        context = ""
        metadata = []
        for i in range(len(res["data"]["Get"][cname])):
            context += "Chunk ID: " + str(chunk_id) + "\n"
            context += res["data"]["Get"][cname][i]["text"] + "\n\n"
            met = res["data"]["Get"][cname][i]["metadata"]
            metadata.append(met.split(".png")[0])
            chunk_id += 1
    except Exception as e:
        print("Exception in DB, dude.")
        print(e)
        if retry_count > 0:
            time.sleep(3)
            context, metadata = qdb(
                query, db_client, name, cname, chunk_id, limit, retry_count - 1
            )
        else:
            print("Retry limit exceeded, returning empty context and metadata.")
            return "There is some error on Vector Database", []
    return context, metadata


# def check_email_exists(email):
#     table_id = "my-project-41692-400512.jobbot.new_users"
#     query = """
#     SELECT *
#     FROM `{}`
#     WHERE email = @email
#     """.format(
#         table_id
#     )

#     job_config = bigquery.QueryJobConfig(
#         query_parameters=[
#             bigquery.ScalarQueryParameter("email", "STRING", email),
#         ]
#     )

#     query_job = dbclient.query(query, job_config=job_config)

#     results = list(query_job.result())
#     return len(results) > 0


def insert_user_data(password, email, userID):
    """
    Insert user data into the BigQuery table.

    Args:
        password (str): User's password.
        email (str): User's email. The case of the email will be preserved exactly as provided.
        userID (str): The unique identifier for the user.

    Note:
        This function expects the 'email' parameter to be provided with its case preserved to ensure
        case sensitivity is maintained in the database.
    """
    table_id = "jobbot-415816.jobBot.users"

    # Data to insert
    rows_to_insert = [
        {
            "password": password,
            "email": email,  # Email is stored exactly as provided, preserving case
            "userID": userID,
            "role": "user",
            "isAnswer": False,
            "iswhiteList": False,
        }
    ]

    # Make an API request to insert the data
    errors = dbclient.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print(f"Encountered errors while inserting rows: {errors}")


def check_username_exists(username, email):
    table_id = "jobbot-415816.jobBot.users"
    query = """
    SELECT * 
    FROM `{}`
    WHERE name = @username and email = @email
    """.format(
        table_id
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("username", "STRING", username),
            bigquery.ScalarQueryParameter("email", "STRING", email),
        ]
    )

    query_job = dbclient.query(query, job_config=job_config)

    results = list(query_job.result())
    return len(results) > 0


def check_email_exists(email):
    """
    Checks if the given email exists in the database, considering case insensitivity.

    Args:
        email (str): The email address to check.

    Returns:
        bool: True if the email exists regardless of case, False otherwise.
    """
    table_id = "jobbot-415816.jobBot.users"
    query = """
    SELECT email 
    FROM `{}`
    WHERE LOWER(email) = LOWER(@email)
    """.format(
        table_id
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email),
        ]
    )

    query_job = dbclient.query(query, job_config=job_config)

    results = list(query_job.result())
    return len(results) > 0


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        password = request.form.get("password")
        email = request.form.get("email")

        # Ensure all fields are provided
        if not password or not email:
            message = "All fields are required."
            return render_template("signup.html", message=message)

        # Check if the username or email already exists

        if check_email_exists(email):
            message = " Email already exists."
            return render_template("signup.html", message=message)

        # Proceed with registration
        userID = str(uuid.uuid4())

        # Assuming insert_user_data is defined elsewhere to insert new user into the database
        insert_user_data(password, email, userID)

        # Redirect to login with success message or to a success page
        return redirect(url_for("login"))

    # If it's a GET request or any other method, just show the registration form
    return render_template("signup.html")


def update_password(email, new_password):
    hashed_password = generate_password_hash(new_password)

    client = dbclient

    table_id = "jobbot-415816.jobBot.users"

    query = f"""
    UPDATE `{table_id}`
    SET password = @new_password
    WHERE email = @email
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("new_password", "STRING", new_password),
            bigquery.ScalarQueryParameter("email", "STRING", email),
        ]
    )

    query_job = client.query(query, job_config=job_config)  # Make an API request.

    query_job.result()

    return "Password updated successfully."


@app.route("/update_password", methods=["GET", "POST"])
def handle_update_password():
    if request.method == "POST":
        # Retrieve email from session instead of form
        email = session.get("email")
        if not email:
            return "Session expired or invalid.", 400

        new_password = request.form.get("new_password")

        response = update_password(email, new_password)
        # Clear the email from session after use
        session.pop("email", None)

        return redirect(
            url_for("login")
        )  # Assuming 'login' is the endpoint for your login page

    return render_template("reset_password.html")


@app.route("/set_session_and_redirect")
def set_session_and_redirect():
    # Assuming 'email' is passed as a query parameter
    email = request.args.get("email")
    if email:
        session["email"] = email
        print(email, "email of session")
        return redirect("/control_panel")  # Redirect to the reset password page
    return "Email required", 400


def get_user_by_email(email):
    dataset_name = "jobbot-415816.jobBot"
    table_name = "users"

    query = """
        SELECT *
        FROM `{0}.{1}`
        WHERE email = @email
    """.format(
        dataset_name, table_name
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )

    query_job = dbclient.query(query, job_config=job_config)

    results = query_job.result()

    for row in results:
        return {
            "email": row.email,
            "password": row.password,
            "userID": row.userID,
            "isAnswer": row.isAnswer,
            "role": row.role,
            "isWhiteList": row.iswhiteList,
        }

    return None


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        adminLastActivity = getAdminLastActivity()
        differ = datetime.now(timezone.utc) - adminLastActivity

        if differ.total_seconds() > 1 * 60:
            updateAdminFlag(False)

        try:
            email = request.form["email"].lower()
            password = request.form["password"]

            if email.lower() == "logout@capria.vc" and password == "XYZZY2024":
                updateAdminFlag(False)
                return render_template(
                    "login.html",
                    message=f"Admin is logged out Now...",
                )

            user = get_user_by_email(email)

            if user and password == user["password"]:

                if getAdminFlag() == True and user["role"] == "admin":
                    return render_template(
                        "login.html",
                        message=f"Admin is already logged In...",
                    )

                if getAdminFlag() == True and user["role"] != "admin":

                    return render_template(
                        "login.html",
                        message=f"Website is under maintenance. Please contact Admin!!",
                    )

                if manage_json("read", "whitelist") == True:
                    print(
                        get_whitelisted_users(user["email"]), "user from whitelisted "
                    )

                    if len(get_whitelisted_users(user["email"])) > 0:
                        session["userID"] = user["userID"]
                        session["role"] = user["role"]
                        session["language"] = "en"
                        session["email"] = user["email"]
                        if user["role"] == "admin":
                            updateAdminFlag(True)
                            time = datetime.now(timezone.utc)
                            updateAdminLastActivity(time)
                        session.modified = True
                        return redirect("/chat")
                    else:
                        return render_template(
                            "login.html",
                            message="You are not allowed to login. Please contact with the admin.",
                        )
                else:
                    session["userID"] = user["userID"]
                    session["role"] = user["role"]
                    session["language"] = "en"
                    session["email"] = user["email"]
                    if user["role"] == "admin":
                        updateAdminFlag(True)
                        time = datetime.now(timezone.utc)
                        updateAdminLastActivity(time)
                        language = get_language(session.get("language", "en"))
                        check_user_exist_all_userList(email, language)

                    session.modified = True
                    return redirect("/chat")

            else:
                return render_template(
                    "login.html",
                    message="Invalid username or password",
                )
        except Exception as e:
            print(e, "error in login")
            return f"Error: {e}", 500
    return render_template("login.html")


def check_user_exist_all_userList(email, language):
    layer1_blankData = {
        "openaiKey": "",
        "layer1URL": "",
        "layer1DriveURL": "",
        "layer1AuthKey": "",
        "classNamelayer1": "",
        "process_image": True,
        "layer1SliderValue": 0,
    }
    layer2_blankData = {
        "openaiKey": "",
        "layer2URL": "",
        "layer2DriveURL": "",
        "layer2AuthKey": "",
        "classNamelayer2": "",
        "process_image": True,
        "layer2SliderValue": 0,
    }
    user_config = {
        "username": email,
        "language": language,
        "projectName": "Enter your Project Name on Control Panel",
        "intro": "Enter your Intro on Control Panel",
        "level2_search_prompt": "Enter your L2 Prompt on Control Panel",
        "l1_text": "Enter your L1 Text on Control Panel",
        "l2_text": "Enter your L2 Text on Control Panel",
        "p1": "Enter your Prompt1 on Control Panel",
        "p2": "Enter your Prompt2 on Control Panel",
        "layer1_config": layer1_blankData,
        "layer2_config": layer2_blankData,
        "vectorFileChange": {"userName": email, "time": "", "layer": ""},
        "openaiKey": "",
        "email":email
    }

    userEmail = get_email_from_username(email)
    if not userEmail:
        save_user_config(email, user_config)


@app.route("/forgot_password", methods=["GET"])
def forgot_password():
    return render_template("forgot_password.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json

    print(session["transcription"], "l1 response from session ")
    try:
        thumbs = data.get("type", "Text")
        l2 = data.get("l2ResponseClicked")
        feedback_text = data.get("feedback", "Thumbs UP/Thumbs Down")
        level = data.get("level", "test")
        question = data.get("question", "")
        l1Response = data.get("L1response", "")
        l2Response = data.get("L2response", "")

        print(thumbs, feedback_text, level)

        add_row_to_sheet(
            [question, feedback_text, thumbs, l1Response, l2Response],
            "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            "dd_Bot",
        )

    except Exception as e:
        update_logs(e)

    return jsonify({"status": "success"})


def transcribe(audio_file):
    try:
        response = lm_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
        transcription = response.text
        language = response.text + " " + response.language
    except Exception as e:
        print(e)
        update_logs(e)
        transcription = "Error."
        language = "english"

    return transcription, language


class FileWithNames(io.BytesIO):
    name = "audio.wav"


def update_logs(input_string):
    file_exists = os.path.isfile("logs.txt")

    with open("logs.txt", "a" if file_exists else "w") as file:
        if file_exists:
            file.write("\n\n\n\n")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{current_time}\n{input_string}\n")


def process_response(input_string, replacements):
    def replacement(match):
        index = int(match.group(1))
        return (
            f"[{replacements[index]}]" if index < len(replacements) else match.group(0)
        )

    try:
        return re.sub(r"\[(\d+)\]", replacement, input_string)
    except:
        return input_string


import requests


def translate_text(text, target_language):
    print(target_language)
    if target_language == "english":
        return text
    api_key = "AIzaSyAtfrkxLhTygIJi9Rb-l0duA8fV9LgKZ7M"  # Replace with your API key

    url = "https://translation.googleapis.com/language/translate/v2"
    data = {"q": text, "target": target_language, "format": "text"}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    response = requests.post(url, headers=headers, params=params, json=data)
    r = response.json()
    print(r)
    return r["data"]["translations"][0]["translatedText"]


def get_language(lang):
    print("getting lang.")
    lang = lang.lower()
    if "arabic" in lang:
        return "ar"
    if "kannada" in lang:
        return "kn"
    if "telugu" in lang:
        return "te"
    if "spanish" in lang:
        return "es"
    if "hebrew" in lang:
        return "he"
    if "japanese" in lang:
        return "ja"
    if "korean" in lang:
        return "ko"
    if "hindi" in lang:
        return "hi"
    if "bengali" in lang:
        return "bn"
    if "tamil" in lang:
        return "ta"
    if "urdu" in lang:
        return "ur"
    if "chinese" in lang:
        return "zh-CN"
    if "french" in lang:
        return "fr"
    if "german" in lang:
        return "de"

    session["language"] = "english"
    return "en"


@app.route("/level1", methods=["POST"])
@login_required
@user_activity_tracker
def level1():
    print("level 1....\n\n\n")
    session["level_1_text"] = (
        session["level_1_text"] if "level_1_text" in session else "Level1:"
    )

    session["transcription"] = request.form["query"] if "query" in request.form else ""
    session["prompt_level1"] = (
        "" if "prompt_level1" not in session else session["prompt_level1"]
    )

    if request.form["leng"] != "":
        session["language"] = request.form["leng"]

    session["language"] = (
        "english" if session["language"] == "" else session["language"]
    )

    audio_file = request.files["audio"] if "audio" in request.files else None

    try:
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = FileWithNames(audio_bytes)
            session["transcription"], session["language"] = transcribe(audio_file)
            session["language"] = get_language(session["language"])
            if session["language"].lower() != "en":
                session["transcription"] = translate_text(
                    session["transcription"], "en"
                )
    except Exception as e:
        session["language"] = "en"
        print(e)
        update_logs(e)
        session["transcription"] = "Error."
    return jsonify({"message": "Data received, start streaming"})


def process_strings_synomums(sentence, synonym_list):
    matches = {}
    for word in sentence:
        for syn_list in synonym_list:
            if word in syn_list:
                matches[word] = syn_list
            else:
                pass
    return matches


def group_sentence(sentence, group_size=3):
    words = sentence.split()  # Split the sentence into words
    # Create groups with a step size of 1, but only include groups that meet the minimum size requirement
    grouped_words = [
        " ".join(words[i : i + group_size])
        for i in range(0, len(words))
        if len(words[i : i + group_size]) == group_size
    ]
    return grouped_words


def get_syn_query_and_reasoning_string(original_sentence, synonyms):
    if synonyms == []:
        return original_sentence, original_sentence

    sentence1 = group_sentence(original_sentence, 1)
    sentence2 = group_sentence(original_sentence, 2)
    sentence3 = group_sentence(original_sentence, 3)
    match1 = process_strings_synomums(sentence1, synonyms)
    match2 = process_strings_synomums(sentence2, synonyms)
    match3 = process_strings_synomums(sentence3, synonyms)
    merged_matches = {**match1, **match2, **match3}
    filterted_syns = []
    query_string = original_sentence

    for key, value in match3.items():
        if key in query_string:
            query_string = query_string.replace(key, " / ".join(value))
            filterted_syns.append(key)

    for key, value in match2.items():
        if key in query_string:
            query_string = query_string.replace(key, " / ".join(value))
            filterted_syns.append(key)

    for key, value in match1.items():
        if key in query_string:
            query_string = query_string.replace(key, " / ".join(value))
            filterted_syns.append(key)

    reasoning_string = original_sentence

    for word in filterted_syns:
        reasoning_string = (
            reasoning_string
            + "\n"
            + "The word {} means the same as the words {}".format(
                word, ",".join(merged_matches[word])
            )
        )

    print("\n\n\n\n\n\n\n")
    print(query_string)
    print(reasoning_string)
    print("\n\n\n\n\n\n\n")
    return query_string, reasoning_string


def update_strings_list(elements_list, strings_list):
    # Loop through each element in the elements list
    for element in elements_list:
        # Check if the element is present in any of the strings in the second list
        if not any(element in string for string in strings_list):
            # If the element is not found in any string, add it to the strings list
            strings_list.append(element)
    return strings_list


from functools import reduce


def remove_characters_from_list(string_list, characters_to_remove):
    # Iterate over each string in the list and remove specified characters
    cleaned_list = [
        reduce(lambda s, char: s.replace(char, ""), characters_to_remove, s)
        for s in string_list
    ]
    return cleaned_list


@app.route("/level1/stream")
@login_required
@user_activity_tracker
def level1_stream():
    global p1, l1_text
    existing_data = readAndWriteJsonData("configu.json", "r")

    layer_1 = existing_data["layer1"]

    print(layer_1, "layer 1  dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    # Load the configuration to get className1
    data = readAndWriteJsonData("configu.json", "r")
    config_data = data.get("layer1", demo_configuration("layer1"))
    # email = session.get(
    #     "current_email", session.get("email")
    # )  # Fallback to default logged-in email
    email = manage_json("read", "instanceRunning")
    print(f"Using email for operations: {email}")  # Debug print

    data_file_path = f"tag_for_each_sentence_{email}.json"
    print(f"Looking for file: {data_file_path}")  # Debug print

    # Construct the file path using the selected user's email

    class_name1 = config_data["classNamelayer1"]
    print(class_name1, "class Name is given")
    # Capitalize the first letter of className1
    ClassName1 = class_name1.capitalize()
    data_file_path = f"tag_for_each_sentence_{email}.json"
    print(ClassName1, "what is this mannnnnnnnn")

    synonyms = manage_json("read", "synonyms")
    query_string, reasoning_string = get_syn_query_and_reasoning_string(
        session["transcription"].lower(), synonyms
    )

    try:
        context1 = ""
        metadata1 = []
        # Use class_name1 for ddbot300
        process_as_image = config_data["process_image"]

        if process_as_image:
            # context1, metadata1 = qdb(
            #     query_string,
            #     layer_1,
            #     class_name1+'large',
            #     ClassName1+'large',
            #     chunk_id=1,
            #     limit=6,
            # )
            # # print(context1, class_name1, ClassName1, "the value is here should we take")
            # context2, metadata2 = qdb(
            #     query_string,
            #     layer_1,
            #     class_name1+'small',
            #     ClassName1+'small',
            #     chunk_id=7,
            #     limit=7,
            # )
            # context = context1 + context2
            # metadata = metadata1 + metadata2
            # context = context1
            # metadata = metadata1
            # Ensure username is retrieved from session

            worddata = read_json_file(data_file_path)
            print("userid ti e chose hereeeeeeee", email)
            tags = ask_gpt_tags(query_string.lower())
            tags = [tag.lower() for tag in tags]
            words = word_tokenize(query_string.lower())
            stoptags = [
                word for word in words if word.lower() not in stopwords.words("english")
            ]
            tags = update_strings_list(stoptags, tags)
            tags = remove_characters_from_list(tags, ["?", ".", "!"])
            # Original part of your function where you might call get_relevant_tags
            most_closest = get_relevant_tags(
                tags, class_name1, ClassName1
            )  # Updated to pass the dynamic class names
            print(class_name1, ClassName1, "checknndndndqkdnndndndndnd")

            context, metadata = get_all_sentences(most_closest, worddata)
            context = join_strings_with_ids(context)
            m = join_strings_with_ids(metadata)
            print(context)
            print(m)

        else:
            print("..")
            context, metadata = qdb(
                query_string,
                layer_1,
                class_name1,
                ClassName1,
                chunk_id=1,
                limit=15,
            )

        sufficient = False
    except Exception as e:
        print("\n\n\nError:    ", e)
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    email = manage_json("read", "instanceRunning")

    try:
        resp = Response(
            ask_gpt(
                question=reasoning_string,
                context=context,
                gpt="gpt-4",
                language=session["language"],
                metadata=metadata,
                addition=manage_json("read", "p1"),
                userid=session["userID"],
                filename=f"Layer_1_file_{email}/file_info_1.json",
                intro_msg=manage_json("read", "l1"),
            ),
            content_type="text/event-stream",
        )
        return resp
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type="text/event-stream")


@app.route("/level2", methods=["POST"])
@login_required
@user_activity_tracker
def level2():
    session["level_2_text"] = (
        session["level_2_text"] if "level_2_text" in session else "Level2:"
    )
    session["layer_1_response"] = request.form["response"]
    session["transcription"] = request.form["query"]
    return jsonify({"message": "Data received, start streaming"})


@app.route("/level2/stream")
@login_required
@user_activity_tracker
def level2_stream():
    global l1_text
    email = manage_json("read", "instanceRunning")
    print("\n\n\nLever 2....\n\n\n")
    data = readAndWriteJsonData("configu.json", "r")
    config_data1 = data.get("layer2", demo_configuration("layer2"))
    class_name2 = config_data1["classNamelayer2"]
    print(class_name2, "class Name is given")
    # Capitalize the first letter of className1
    ClassName2 = class_name2.capitalize()
    print(ClassName2, "what is this mannnnnnnnn")
    global p2, layer_2
    print(layer_2, "layer_2 is working...................................")
    synonyms = manage_json("read", "synonyms")
    query_string, reasoning_string = get_syn_query_and_reasoning_string(
        session["transcription"].lower(), synonyms
    )
    query_string = session["transcription"].lower()
    reasoning_string = session["transcription"].lower()
    try:
        context, metadata = qdb(
            query_string,
            layer_2,
            class_name2,
            ClassName2,
            chunk_id=0,
            limit=5,
        )
        sufficient = False
        print(context)
        print(metadata)
    except Exception as e:
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    session["response"] = session["layer_1_response"]

    try:
        resp = Response(
            ask_gpt(
                question=reasoning_string,
                context=context,
                gpt="gpt-4",
                language=session["language"],
                metadata=metadata,
                addition=manage_json("read", "p2"),
                userid=session["userID"],
                filename=f"Layer_2_file_{email}/file_info_1.json",
                intro_msg=manage_json("read", "l2"),
            ),
            content_type="text/event-stream",
        )

        return resp
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type="text/event-stream")


def insert_data(table_name, data):
    table_id = f"jobbot-415816.jobBot.{table_name}"

    if not isinstance(data, list):
        data = [data]

    errors = dbclient.insert_rows_json(table_id, data)

    if errors == []:
        print(f"Data added successfully into {table_name}")
    else:
        print(f"Encountered errors while inserting into {table_name} : {errors}")


def delete_data(table_name, identifier_column, identifier_value):

    table_id = f"jobbot-415816.jobBot.{table_name}"

    sql_query = f"""
        DELETE FROM `{table_id}`
        WHERE `{identifier_column}` = '{identifier_value}'
    """

    query_job = dbclient.query(sql_query)  # Make an API request.
    query_job.result()  # Waits for the query to finish

    print(
        f"Rows deleted in {table_name} where {identifier_column} is {identifier_value}."
    )


@app.route("/users")
@admin_required
@user_activity_tracker
def users():
    table_id = "jobbot-415816.jobBot.users"

    sql_query = """
    SELECT * 
    FROM `{}`
    """.format(
        table_id
    )

    query_job = dbclient.query(sql_query)
    results = query_job.result()
    user_data = []
    for row in results:
        user_data.append(dict(row))

    return jsonify(user_data)


from google.api_core.exceptions import BadRequest


def update_user_is_answered(userID, retry_count=3):
    print(userID, "user updated")
    try:
        # Define the update query
        query = f"""
        UPDATE `jobbot-415816.jobBot.users`
        SET isAnswer = true
        WHERE userID = '{userID}'
        """

        # Execute the query
        query_job = dbclient.query(query)
        query_job.result()  # Wait for the query to complete
        print("user isAnswered updated")

    except BadRequest as e:
        # Handle specific error related to the streaming buffer
        if "streaming buffer" in str(e) and retry_count > 0:
            print(
                f"Streaming buffer issue encountered. Retrying... ({retry_count} attempts left)"
            )
            # Retry the operation after a short delay
            time.sleep(20)
            update_user_is_answered(userID, retry_count - 1)
        else:
            # Retry count exceeded or other error occurred
            print(f"An error occurred while updating user isAnswered: {str(e)}")


def update_answer(userID, answerID, answer):
    # Define the update query
    query = f"""
    UPDATE `jobbot-415816.jobBot.answers`
    SET answer = '{answer}'
    WHERE userID = '{userID}' AND answerID = '{answerID}'
    """

    # Execute the query
    query_job = dbclient.query(query)
    query_job.result()  # Wait for the query to complete
    print("Answer updated successfully")


import time


CREDENTIALS_FILE = "drive.json"

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "drive.json"
DOWNLOAD_DIRECTORY = "download_files"
INFO_FILE_PATH = os.path.join(DOWNLOAD_DIRECTORY, "file_info.json")


# Replace with your actual folder ID

########################################################

lm_client = None


layer_1 = None

global layer_2
layer_2 = None

global openai_flag
openai_flag = True

global layer_1_flag
layer_1_flag = True

global layer_2_flag
layer_2_flag = True

global error_admin_msg
error_admin_msg = "error."


@app.route("/loading_status")
@login_required
@user_activity_tracker
def loading_status():
    loading_status = progress_log("", mode="read")
    data = manage_json("read", "vectorFileChange")

    status = {"status": "..."}
    if data == "Field not found.":
        status = {"status": loading_status}
    else:
        status = {"status": loading_status, "vectorFileChange": data}

    return jsonify(status)


def demo_configuration(layer_name):
    # key
    layer_url_key = f"{layer_name}URL"
    layer_drive_url_key = f"{layer_name}DriveURL"
    layer_auth_key = f"{layer_name}AuthKey"
    layer_class_name_key = f"className{layer_name}"

    return {
        "openaiKey": "",
        layer_url_key: "",
        layer_drive_url_key: "",
        layer_auth_key: "",
        layer_class_name_key: "",
    }


def initiate_clients():
    global openai_flag, layer_1_flag, layer_2_flag, lm_client, layer_1, layer_2, error_admin_msg, loading_status
    # config_data = load_configuration()
    data = readAndWriteJsonData("configu.json", "r")
    config_data = data.get("layer1", demo_configuration("layer1"))
    config_data1 = data.get("layer2", demo_configuration("layer2"))

    try:
        openai.api_key = data["openaiKey"]
        lm_client = openai.OpenAI(api_key=data["openaiKey"])
        msg = [
            {"role": "system", "content": "system_message"},
            {"role": "user", "content": "user_message"},
        ]

        response = lm_client.chat.completions.create(
            model="gpt-4",
            messages=msg,
            max_tokens=1000,
            temperature=0.0,
        )
        openai_flag = False
        loading_status = "LLM Client Working..."
        print("lm_client working")
    except:
        openai_flag = True
        print("lm_client not working")
        loading_status = "LLM Client not Working..."
        progress_log(loading_status)

    try:
        layer_1 = weaviate.Client(
            url=config_data["layer1URL"],
            auth_client_secret=weaviate.AuthApiKey(
                api_key=config_data["layer1AuthKey"]
            ),
            additional_headers={"X-OpenAI-Api-Key": data["openaiKey"]},
        )
        print("layer 1 working", layer_1)
        loading_status = "layer 1 working..."
        # check_for_updates(config_data, "download_files", "file_info_1.json")
        layer_1_flag = False
    except:
        layer_1_flag = True

    try:
        layer_2 = weaviate.Client(
            url=config_data1["layer2URL"],
            auth_client_secret=weaviate.AuthApiKey(
                api_key=config_data1["layer2AuthKey"]
            ),
            additional_headers={"X-OpenAI-Api-Key": data["openaiKey"]},
        )

        layer_2_flag = False

        loading_status = "layer 2 working..."
        print("layer 2 working")
        progress_log(loading_status)

    except:
        layer_2_flag = True

    loading_status = None
    error_admin_msg = ""
    if openai_flag:
        error_admin_msg += "Please check your OpenAI API KEY.\n"
    if layer_1_flag:
        error_admin_msg += "Please check your Layer_1 API KEY.\n"
    if layer_2_flag:
        error_admin_msg += "Please check your Layer_2 API KEY.\n"

    if error_admin_msg == "":
        loading_status = "Ready to use..."
        print("layer 2 working")
        progress_log(loading_status)
    else:
        progress_log(error_admin_msg)


SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "drive.json"
TOKEN_FILE = "token.json"


def get_google_drive_service():
    creds = None
    # Check if the token file exists and load the credentials
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Try to refresh the token if it's expired and there is a refresh token
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            # If the refresh token is invalid, delete the token file and set creds to None
            os.remove(TOKEN_FILE)
            creds = None

    # If no valid credentials are available, initiate a new login flow
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    return service


def download_folder(folder_url, save_directory, json_file):
    global loading_status
    loading_status = "Drive Files downloading Started..."
    progress_log(loading_status)

    folder_id = folder_url.split("/")[-1]
    service = get_google_drive_service()

    query = f"'{folder_id}' in parents"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_info_list = []
    page_token = None

    print("____________>>>>>>>>>> download files")

    try:
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items = response.get("files", [])
            for item in items:
                file_id = item["id"]
                file_name = item["name"]
                modified_time = item["modifiedTime"]
                mimeType = item["mimeType"]
                file_path = os.path.join(save_directory, file_name)

                if mimeType.startswith("application/vnd.google-apps."):
                    continue

                drive_request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, drive_request)
                done = False
                print(downloader, "files downlaodfer vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                while not done:
                    _, done = downloader.next_chunk()
                fh.seek(0)

                with open(file_path, "wb") as f:
                    f.write(fh.read())
                    print(f"{file_name} has been downloaded.")
                    loading_status = f"{file_name} has been downloaded."
                    progress_log(loading_status)

                file_info_list.append(
                    {"id": file_id, "name": file_name, "modifiedTime": modified_time}
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        info_file_path = os.path.join(save_directory, json_file)
        with open(info_file_path, "w") as json_file:
            json.dump(file_info_list, json_file, indent=4)
            print("JSON file written successfully.")
            loading_status = "JSON file written successfully."
            progress_log(loading_status)

        loading_status = "Files Downloaded Successfully"
        progress_log(loading_status)
        return {
            "status": "success",
            "message": "Files downloaded successfully.",
            "files": file_info_list,
        }

    except Exception as e:
        print(f"An error occurred while downloading ----------------------------: {e}")
        return {"status": "failure"}


def delete_vectors(dbclient, class_name, search):

    result = dbclient.batch.delete_objects(
        class_name=class_name,
        where={
            "operator": "Like",
            "path": ["metadata"],
            "valueText": search,
        },  # same where operator as in the GraphQL API
        output="verbose",
        dry_run=False,
    )
    return


def count_files_in_directory(directory_path):
    file_count = 0
    for entry in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, entry)):
            file_count += 1
    return file_count


import difflib


def find_closest_match(word, lists_of_synonyms, similarity_threshold=0.7):
    closest_match = None
    closest_list = None
    highest_ratio = 0
    for synonym_list in lists_of_synonyms:
        for synonym in synonym_list:
            ratio = difflib.SequenceMatcher(None, word, synonym).ratio()
            if ratio > highest_ratio and ratio >= similarity_threshold:
                highest_ratio = ratio
                closest_match = synonym
                closest_list = synonym_list

    if closest_list:
        return closest_list
    else:
        return []


def process_synonym_file(data):
    list_of_lists = []
    try:
        lines = data.strip().split("\n")
        list_of_lists = [line.split(" = ") for line in lines]
    except:
        pass
    return list_of_lists


@app.route("/receive_synonym_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file and file.filename.endswith(".txt"):
        content = file.read().decode("utf-8").lower()
        print(content)
        synonym_list = process_synonym_file(content)
        manage_json("write", "synonyms", value=synonym_list)
        return redirect(url_for("control_panel"))
    else:
        return "Invalid file format, please upload a .txt file"


def add_dict_to_json_file(filename, new_data):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data.update(new_data)

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


@app.route("/check_for_updates")
@admin_required
@user_activity_tracker
def check_for_updates_route():
    global layer_1, layer_2, loading_status
    print("vec vec vec")
    layer = request.args.get("layer")
    email = manage_json("read", "instanceRunning")

    if layer not in ["layer1", "layer2"]:
        return (
            jsonify(
                {"error": "Invalid layer specified. Please use 'layer1' or 'layer2'."}
            ),
            400,
        )

    loading_status = "Checking updates."
    progress_log(loading_status)
    data = readAndWriteJsonData("configu.json", "r")
    if not data:
        data = {}

    if layer == "layer1":
        config = data.get("layer1", demo_configuration("layer1"))
        directory = f"Layer_1_file_{email}"
        info_file_name = "file_info_1.json"
        dbclient = layer_1
    else:
        config = data.get("layer2", demo_configuration("layer2"))
        directory = f"Layer_2_file_{email}"
        info_file_name = "file_info_1.json"
        dbclient = layer_2

    layer_config = {
        "driveURL": config.get(f"{layer}DriveURL", ""),
        "layerURL": config.get(f"{layer}URL", ""),
        "layerAuthKey": config.get(f"{layer}AuthKey", ""),
        "className": config.get(f"className{layer}", ""),
        "openaiKey": data.get("openaiKey", ""),
        "process_image": config.get("process_image", True),
    }

    ensure_directories_exist(directory, f"{directory}_all_files")
    try:
        update_result = check_for_updates(
            layer_config, directory, info_file_name, dbclient, email
        )
        manage_json(
            "write",
            "vectorFileChange",
            {
                "userName": session["email"],
                "time": datetime.now().isoformat(),
                "layer": layer,
            },
        )
        if isinstance(update_result, dict) and update_result.get("status") == "success":
            print("\n\n\nUpdate recognized file changes!..\n\n\n")

            modified_files, newly_aded_files = separate_files(directory)

            print(modified_files, "\n  modified files \n")

            print(newly_aded_files, "\n  newely added fileds  \n")

            print(directory, "directory ---------")

            difference = compare_pdf_pages(directory, f"{directory}_all_files")

            print(difference, "difference in both the files----------- ")

            if (
                config["process_image"]
                and count_files_in_directory(directory) > 1
                and len(difference) > 0
                or len(newly_aded_files) > 0
            ):
                print("\n\nImage\n\n")
                img2txt_file = f"image2txt_updates_{email}.json"
                smart_chunk_json = f"smart_chunk_updates_{email}.json"
                filename = f"allsentences_{email}.json"
                tags_file = f"tag_for_each_sentence_{email}.json"
                fliped_data = read_json_file(f"flipped_dict_{email}.json")

                img2txt_file = create_blank_json_with_timestamp(img2txt_file)
                smart_chunk_json = create_blank_json_with_timestamp(smart_chunk_json)
                loading_status = "processing files."

                if len(newly_aded_files) > 0:
                    process_newly_added_pdfs(
                        directory, data.get("openaiKey", ""), img2txt_file
                    )

                    print(fliped_data, "-------------->>>>>>>>>>  flip_data")

                    past_filpped_value = flip_dict_value_and_key(fliped_data)

                    loading_status = "Smart chunking...."
                    print(smart_chunk_json)

                    smart_chunk(img2txt_file, smart_chunk_json)

                    smart_chunk_data = read_json_file(smart_chunk_json)

                    print("\n\n\n\n\n", smart_chunk_data, "\n\n\n\n\n\n")

                    new_smart_chunk = {**smart_chunk_data, **past_filpped_value}

                    add_dict_to_json_file(filename, new_smart_chunk)

                    wholedata = read_json_file(filename)

                    print("\n\n\n\n\n", wholedata, "\n\n\n\n\n\n")

                    flip_dict_and_update_as_json(
                        wholedata, f"flipped_dict_{email}.json"
                    )

                    tag_for_each_sentence(smart_chunk_data, tags_file)

                    tags = read_json_file(tags_file)
                    tags = get_unique_words_case_insensitive(tags)

                if len(modified_files) > 0:
                    process_modified_pdfs(
                        directory, data.get("openaiKey", ""), img2txt_file, difference
                    )

                    updated_fliped_data, removed_data = filter_data(
                        fliped_data, difference
                    )

                    text2Imgdata = read_json_file(img2txt_file)

                    print(updated_fliped_data, "updated  filiped data --------->>>>>")

                    print(removed_data, "removed_ data --------->>>>>")

                    print(text2Imgdata, "-------------->>>>>>>>>>  text2Imgdata")

                    past_filpped_value = flip_dict_value_and_key(updated_fliped_data)

                    save_json_to_directory({}, "./", f"flipped_dict_{email}.json")

                    loading_status = "Smart chunking...."
                    print(smart_chunk_json)

                    smart_chunk(img2txt_file, smart_chunk_json)

                    smart_chunk_data = read_json_file(smart_chunk_json)

                    print("\n\n\n\n\n", smart_chunk_data, "\n\n\n\n\n\n")

                    new_smart_chunk = {**smart_chunk_data, **past_filpped_value}

                    add_dict_to_json_file(filename, new_smart_chunk)

                    wholedata = read_json_file(filename)

                    print("\n\n\n\n\n", wholedata, "\n\n\n\n\n\n")

                    flip_dict_and_update_as_json(
                        wholedata, f"flipped_dict_{email}.json"
                    )

                    tag_for_each_sentence(smart_chunk_data, tags_file)

                    tags = read_json_file(tags_file)
                    tags = get_unique_words_case_insensitive(tags)
                    print(tags, type(tags), "-----------tags and tag type \n \n")
                    class_name_for_vectorization = layer_config["className"]
                    vectorize_tags(tags, class_name_for_vectorization)
            elif count_files_in_directory(directory) > 1 and len(difference) > 0:
                print("\n\nNot Image\n\n")

                pdf_vectorization(directory, layer_config)
            else:
                print("\n\nNo update recognised\n\n")
                loading_status = "No files updated."
                progress_log(loading_status)

            print("Files re-vectorized successfully.")
        else:
            return (
                jsonify(
                    {"error": "Failed to process files.", "details": update_result}
                ),
                500,
            )
    except Exception as e:
        app.logger.error(f"Error while processing files: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(directory):
            copy_files(directory, f"{directory}_all_files")
            for file_name in os.listdir(directory):
                if not file_name.startswith("file_info"):
                    file_path = os.path.join(directory, file_name)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        print(f"Deleted {file_name}")
                    except Exception as e:
                        app.logger.error(f"Failed to delete {file_name}. Reason: {e}")
            print("Non-essential files deleted after vectorization.")
    loading_status = "Ready to use..."
    progress_log(loading_status)
    return jsonify({"message": "Files processed successfully."})


def delete_keys_with_keyword(filename, keyword):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
        return []
    except json.JSONDecodeError:
        print(f"The file {filename} is not a valid JSON file or is empty.")
        return []

    deleted_values = []

    keys_to_delete = [key for key in data.keys() if keyword in key]

    for key in keys_to_delete:
        deleted_values.append(data[key])
        del data[key]

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    return deleted_values


def separate_files(directory1):
    modified = []
    newly_added = []

    files_dir1 = os.listdir(directory1)

    for file_name in files_dir1:
        if file_name.startswith("modi_"):
            clean_name = file_name[5:]
            modified.append(clean_name)
        elif not file_name.startswith("file_info"):
            newly_added.append(file_name)

    return modified, newly_added


def compare_pdf_pages(directory1, directory2):
    differences = []
    new_files = {
        f[5:]
        for f in os.listdir(directory1)
        if f.endswith(".pdf") and f.startswith("modi_")
    }

    print(directory1, directory2, "direct 1 and 2------")
    old_files = {f for f in os.listdir(directory2) if f.endswith(".pdf")}

    print(f"{new_files} new files {old_files} old files")
    common_files = new_files.intersection(old_files)

    print(common_files, "common files")

    for file in common_files:
        new_file_path = os.path.join(directory1, f"modi_{file}")
        old_file_path = os.path.join(directory2, file)

        new_pdf = old_pdf = None

        try:
            new_pdf = fitz.open(new_file_path)
            old_pdf = fitz.open(old_file_path)
            num_pages = min(new_pdf.page_count, old_pdf.page_count)
            pages_differ = []

            for i in range(num_pages):
                new_page = new_pdf.load_page(i)
                old_page = old_pdf.load_page(i)

                new_text = new_page.get_text()
                old_text = old_page.get_text()
                if new_text != old_text:
                    pages_differ.append(i + 1)

            if new_pdf.page_count > old_pdf.page_count:
                pages_differ.extend(range(num_pages + 1, new_pdf.page_count + 1))

            if pages_differ:
                differences.append({f"modi_{file}": pages_differ})

        except Exception as e:
            print(f"Error processing {file}: {e}")

        finally:
            if new_pdf:  # Check if object is instantiated before trying to close
                new_pdf.close()
            if old_pdf:
                old_pdf.close()

    return differences


def check_for_updates(layer_config, directory, info_file_name, dbclient, email):
    drive_url = layer_config["driveURL"]
    img_process = layer_config["process_image"]
    classname = layer_config["className"]

    loading_status = "Looking in Drive..."

    info_file_path = os.path.join(directory, info_file_name)

    service = get_google_drive_service()
    folder_id = drive_url.split("/")[-1]
    query = f"'{folder_id}' in parents"

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        existing_files_info = []
        if os.path.exists(info_file_path):
            with open(info_file_path, "r") as file:
                existing_files_info = json.load(file)

        existing_files_ids = {file["id"]: file for file in existing_files_info}

        current_files_info = []
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items = response.get("files", [])

            print(items, "items form gdrive ---------")

            for item in items:
                current_files_info.append(item)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        files_to_be_deleted = []

        for item in current_files_info:
            modified, _ = process_file(service, item, directory, existing_files_ids)
            if modified:
                files_to_be_deleted.append(item["name"])

        files_to_be_deleted = files_to_be_deleted + detect_and_handle_deleted_files(
            existing_files_ids, current_files_info, directory
        )

        print(files_to_be_deleted, "---------------files to be deleted")

        for filename in files_to_be_deleted:
            filename = filename.split(".pdf")[0]
            filename = filename.split(".docx")[0]
            print(filename, "filename ---------------")
            loading_status = "Deleting {}".format(filename)
            progress_log(loading_status)

            if img_process:
                # delete_file("flipped_dict.json" )
                sentences_deleted = delete_keys_with_keyword(
                    f"allsentences_{email}.json", filename.split(".pdf")[0]
                )

                tags_file_name = f"tag_for_each_sentence_{email}.json"

                # for sentence in sentences_deleted:
                #     delete_keys_with_keyword(f"tag_for_each_sentence_{email}.json", sentence)

                wholedata = read_json_file(f"allsentences_{email}.json")

                print(wholedata, "whole data ------------           \n \n ")

                flip_dict_and_store_as_json(wholedata, f"flipped_dict_{email}.json")

                tag_for_each_sentence_after_delete(wholedata, tags_file_name)

            else:
                delete_vectors(dbclient, classname, filename)
                pass

        with open(info_file_path, "w") as json_file:
            json.dump(current_files_info, json_file, indent=4)
            loading_status = f"File info JSON updated for {directory}."
            progress_log(loading_status)
            print(f"File info JSON updated for {directory}.")

        return {"message": "Update  check completed successfully", "status": "success"}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"message": str(e), "status": "error"}, 500


def process_file(service, item, directory, existing_files_ids):
    global loading_status
    loading_status = "Loading or Updating Starting..."

    file_id = item["id"]
    file_name = item["name"]
    mimeType = item["mimeType"]
    modified_time = item["modifiedTime"]
    file_path = os.path.join(directory, file_name)

    folder_name = os.path.dirname(file_path)

    if file_name.startswith("~$"):
        print(f"Skipping temporary or system file: {file_name}")
        return False, None

    file_is_new_or_updated = False

    if file_id not in existing_files_ids:
        download_file(service, file_id, file_path, mimeType, file_name)
        print(f"Downloaded file: {file_name}")

        file_is_new_or_updated = True

    if (file_id in existing_files_ids) and (
        existing_files_ids[file_id].get("modifiedTime", "") != modified_time
    ):
        modi_file_name = f"modi_{file_name}"
        modi_file_path = os.path.join(directory, modi_file_name)
        download_file(service, file_id, modi_file_path, mimeType, modi_file_name)
        print(f"Downloaded file: {file_name}")
        file_is_new_or_updated = True

    return file_is_new_or_updated, file_path


def save_file(directory, filename, content):

    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, filename)

    with open(file_path, "w") as file:
        file.write(content)
        print(f"File saved: {file_path}")


def detect_and_handle_deleted_files(existing_files_info, current_files_info, directory):
    current_files_ids = {file["id"] for file in current_files_info}
    files_to_be_deleted = []
    print(f"Current files in Drive: {current_files_ids}")
    for existing_file_ids, file_data in existing_files_info.items():
        if existing_file_ids not in current_files_ids:
            files_to_be_deleted.append(file_data["name"])

    return files_to_be_deleted


def download_file(service, file_id, file_path, mimeType, file_name):
    try:
        if mimeType.startswith("application/vnd.google-apps."):
            export_mime_type = "application/pdf"
            file_extension = "pdf"
            if mimeType == "application/vnd.google-apps.document":
                export_mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                file_extension = "docx"
            elif mimeType == "application/vnd.google-apps.spreadsheet":
                export_mime_type = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                file_extension = "xlsx"
            elif mimeType == "application/vnd.google-apps.presentation":
                export_mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                file_extension = "pptx"
            file_path = f"{file_path.rsplit('.', 1)[0]}.{file_extension}"
            request = service.files().export_media(
                fileId=file_id, mimeType=export_mime_type
            )
        else:
            request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        with open(file_path, "wb") as f:
            f.write(fh.getvalue())
            print(f"Downloaded or exported file: {file_name} to {file_path}")

    except Exception as e:
        print(f"An error occurred while downloading/exporting '{file_name}': {e}")


def split_pdf_text_by_page(pdf_path):
    pages = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            pages.append(text)
    return pages


def load_documents(directory, glob_patterns):
    documents = []
    for glob_pattern in glob_patterns:
        file_paths = glob.glob(os.path.join(directory, glob_pattern))
        for fp in file_paths:
            print(fp)
            try:
                if fp.endswith(".docx"):
                    text = docx2txt.process(fp)
                    pages = [text]  # Treat the whole document as a single "page"
                    print("Done processing Docs.")
                elif fp.endswith(".pdf"):
                    pages = split_pdf_text_by_page(fp)
                    print("Done processing PDFs.")
                else:
                    print(f"Warning: Unsupported file format for {fp}")
                    continue
                documents.extend(
                    [
                        (page, os.path.basename(fp), i + 1)
                        for i, page in enumerate(pages)
                    ]
                )
                print("ADDED ", fp)
            except Exception as e:
                print(f"Warning: The file {fp} could not be processed. Error: {e}")
    return documents


def split_text(text, file_name, chunk_size, chunk_overlap):
    start = 0
    end = chunk_size
    while start < len(text):
        yield (text[start:end], file_name)
        start += chunk_size - chunk_overlap
        end = start + chunk_size


def split_documents(documents, chunk_size, chunk_overlap):
    texts = []
    metadata = []
    for doc_text, file_name, page_number in documents:
        for chunk in split_text(doc_text, file_name, chunk_size, chunk_overlap):
            sentence = chunk[0]
            texts.append(sentence)
            metadata.append(str(file_name) + " Pg: " + str(page_number))

    return texts, metadata


def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def convert_pdf2img(input_file, output_dir="eximages"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_directory(output_dir)

    pdfIn = fitz.open(input_file)
    output_files = []
    for pg in range(pdfIn.page_count):
        page = pdfIn.load_page(pg)
        zoom_x = 2
        zoom_y = 2
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(input_file))[0]}_page_{pg+1}.png",
        )
        pix.save(output_file)
        output_files.append(output_file)
    pdfIn.close()
    return output_files


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_text_from_image(image_path, api_key, file_path):
    base64_image = encode_image(image_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The image is from a PPT. Traditional text extraction does not preserve the structure. In that, I can still extract the text, but, it will be impossible to format it into correct headings and sections, and questions and asnwers, as it is not a tradtional document. I want you to return properly formatted text. Do not skip out on anything. Also, do not say 'here is what the image says..', etc. Directly start producing image transcription .I will be directly copying this into a document. people should not know this has been generated by you. People reading it must be convinced and it should carry the same and complete information as in the image. Include extremly minute details. The structure of the data must be exactly as seen on the slide. Do Not try to skip any details. include every single detail in a manner that represents exactly what the image is trying to convey.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4000,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        response_data = response.json()
        text_response = (
            response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
    else:
        text_response = f"Error: {response.status_code}, Message: {response.text}"

    filename = image_path.replace("eximages", "")
    new_data = {filename: text_response}
    loading_status = f"Processed {filename}"
    progress_log(loading_status)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data.update(new_data)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def process_all_pdfs(pdf_folder, api_key, file_path, fileAndPageData=None):
    delete_files("eximages")
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            convert_pdf2img(pdf_path)
            loading_status = f"Processed {filename}"
            progress_log(loading_status)
            print(f"\n\nProcssing {filename}\n\n")
            for image_filename in os.listdir("eximages"):
                image_path = os.path.join("eximages", image_filename)
                print(image_path)
                get_text_from_image(image_path, api_key, file_path)


def process_modified_pdfs(pdf_folder, api_key, file_path, page_data):
    delete_files("eximages")

    print(r"\n\process_modified_pdfs\n\n")

    if len(page_data) <= 0:
        return

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf") and filename.startswith("modi_"):
            pdf_path = os.path.join(pdf_folder, filename)

            pagesArray = consolidate_page_data(page_data, filename)

            if len(pagesArray):
                convert_modified_pdf2img(pdf_path, pagesArray)

            loading_status = f"Processed {filename}"
            progress_log(loading_status)
            print(f"\n\nProcessing {filename}\n\n")

            for image_filename in os.listdir("eximages"):
                image_path = os.path.join("eximages", image_filename)
                print(image_path)
                get_text_from_image(image_path, api_key, file_path)


def process_newly_added_pdfs(pdf_folder, api_key, file_path):
    print(file_path, "file path \n")
    for filename in os.listdir(pdf_folder):
        print(filename, "filenames ----------- \n \n")
        if filename.endswith(".pdf") and not filename.startswith("modi_"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(pdf_path, "pdf path \n \n -------")
            convert_pdf2img(pdf_path)
            loading_status = f"Processed {filename}"
            progress_log(loading_status)
            print(f"\n\nProcessing {filename}\n\n")

            for image_filename in os.listdir("eximages"):
                print(image_filename, "image filename \n")
                image_path = os.path.join("eximages", image_filename)
                print(image_path)
                get_text_from_image(image_path, api_key, file_path)


def convert_modified_pdf2img(input_file, pages_to_convert, output_dir="eximages"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_directory(output_dir)

    pdfIn = fitz.open(input_file)
    output_files = []
    total_pages = pdfIn.page_count

    print(
        f"{input_file} input_file {pages_to_convert} pages_to_convert ------------- ::::::::::::"
    )

    pages_to_process = [
        page - 1 for page in pages_to_convert[0] if 0 < page <= total_pages
    ]

    for pg in pages_to_process:
        page = pdfIn.load_page(pg)
        zoom_x, zoom_y = 2, 2
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        base_filename = os.path.basename(input_file)
        if base_filename.startswith("modi_"):
            base_filename = base_filename[5:]  # Remove the 'modi_' prefix

        output_file = os.path.join(
            output_dir, f"{os.path.splitext(base_filename)[0]}_page_{pg + 1}.png"
        )
        pix.save(output_file)
        output_files.append(output_file)

    pdfIn.close()
    return output_files


def filter_data(data, filters):

    pages_to_remove = set()
    removed_data = {}

    # Create a set of pages to remove based on filters
    for filter_dict in filters:
        for filename, pages in filter_dict.items():
            stripped_filename = re.sub(r"^modi_(.*)\.pdf$", r"\1", filename)
            for page in pages:
                print(page)
                page_key = f"{stripped_filename}_page_{page}.png"
                pages_to_remove.add(page_key)

    # Filter the data
    for key, value in list(data.items()):

        if any(page in value for page in pages_to_remove):
            print(
                page,
            )
            removed_data[key] = value
            del data[key]

    return data, removed_data


def consolidate_page_data(page_data, filename):
    consolidated_data = {}
    for entry in page_data:
        for file_key, page_number in entry.items():
            if file_key == filename:
                if filename in consolidated_data:
                    consolidated_data[filename].append(page_number)
                else:
                    consolidated_data[filename] = [page_number]

    return consolidated_data[filename]


def pdf_vectorization(directory, layer_config, processed_files=set(), chunk_size=400):
    global loading_status
    class_name = layer_config["className"]
    url = layer_config["layerURL"]
    auth_key = layer_config["layerAuthKey"]
    data = readAndWriteJsonData("configu.json", "r")
    openai_key = data["openaiKey"]
    print(class_name, "sssssssssssssssssssssssssssssssssssssssssssssssssssssss")

    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=auth_key),
        additional_headers={"X-OpenAI-Api-Key": openai_key},
    )
    loading_status = "Vectorizing Documents..."
    progress_log(loading_status)
    print(client, "client is the value of i am getting")
    glob_patterns = ["*.docx", "*.pdf"]

    print("\n\n\n\nLoading documents.\n\n\n\n")
    documents = load_documents(directory, glob_patterns)
    print("\n\n\n\nDone Loading documents.\n\n\n\n")
    chunk_overlap = 0
    texts, metadata = split_documents(documents, chunk_size, chunk_overlap)
    print("\n\n\n\nDone Splitting documents.\n\n\n\n")

    data_objs = [{"text": tx, "metadata": met} for tx, met in zip(texts, metadata)]
    total = len(data_objs)

    i = 0

    class_obj = {
        "class": class_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "metadata",
                "dataType": ["text"],
            },
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text",
            },
        },
    }
    try:
        client.schema.create_class(class_obj)
    except Exception as e:
        print("Error:", e)
        print("--------*--------**")
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

    client.batch.configure(batch_size=50)
    with client.batch as batch:
        print("Batch=========================")
        for data_obj in data_objs:
            # filename = os.path.basename(data_obj["metadata"])
            # if filename not in processed_files:
            i += 1
            print(i)

            if i > -1:
                loading_status = "Uploaded: " + str(i) + "/" + str(total)

                print(loading_status)
                progress_log(loading_status)

                batch.add_data_object(data_obj, class_name)
                # processed_files.add(filename)  # Mark the file as processed
            else:
                print("Already present.", i)

    res = (
        client.query.get(class_name, ["text", "metadata"])
        .with_near_text({"concepts": "What do I do with my life??"})
        .with_limit(10)
        .do()
    )
    print("\n\n", class_name, "\n\n")
    print(res)
    return processed_files


smart_function = [
    {
        "name": "return_response",
        "description": "to be used to return list of chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of chunks directly extracted from the document given",
                    "items": {"type": "integer"},
                },
            },
            "required": ["item_list"],
        },
    }
]


def ask_gpt_smart_chunk(question):
    system_message = "You are a smart chunker. You will be given content from a slide/document page. You need to return the data, but divided into chunks - meaning, the chunk you return must encapsulate complete information. You are allowed to return as many chunks as you like. But, you must cover the entire information. The reasons is that I will be feeding this into a vector database for semantic retrival of vectors. By feeding an entire page, the similarity scores are very low for specific queries that are only a fraction of the larger page. But, if I were to auto chunk it by 100 or some words, then there are cases where information could be cut off, etc, Therefore, you must return a list of strings. This is called smart chunking. Finally, remeber, what you return, when read, must preserve context. Just returning names, or sentences without any indication of the context or what they represent will be useless. Each chunk must represent the heading it was pulled from, so that when we look at it we know exactly the central context from where it was derived from. it should be so good that a reader must be able to put back the original text be peicing the chunks together. that is how good it should be."
    user_message = "content from page from document below: \n" + question
    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    print("-----------------------")
    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        max_tokens=500,
        temperature=0.0,
        functions=smart_function,
        function_call="auto",
    )
    reply = response.choices[0].message.content
    try:
        reply = ast.literal_eval(reply)
    except:
        try:
            reply = json.loads(response.choices[0].message.function_call.arguments)[
                "item_list"
            ]
            print(reply)
        except:
            print(reply)
            reply = []

    return reply


def smart_chunk(exisiting_json, processed_json_name):
    with open(exisiting_json, "r") as file:
        data = json.load(file)
    new_data = {}

    for key, value in data.items():
        print(key)
        chunk_id = 0
        check = key + " " + str(chunk_id)
        loading_status = "Chunking " + check
        progress_log(loading_status)

        try:
            with open(processed_json_name, "r") as file:
                data = json.load(file)
            if check in data:
                print("Moving on..")
                continue
        except Exception as e:
            print(e)
            data = None
        processed_list = ask_gpt_smart_chunk(value)
        for item in processed_list:
            new_data[key + " " + str(chunk_id)] = item
            if data is not None:
                data.update(new_data)
                new_data = data
            chunk_id += 1
            with open(processed_json_name, "w") as new_file:
                json.dump(new_data, new_file, indent=4)


def json_vectorize(json_file, layer_config, large=True):
    large = "large" if large else "small"
    class_name = layer_config["className"] + large
    url = layer_config["layerURL"]
    auth_key = layer_config["layerAuthKey"]
    openai_key = layer_config["openaiKey"]

    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=auth_key),
        additional_headers={"X-OpenAI-Api-Key": openai_key},
    )

    with open(json_file, "r") as file:
        data = json.load(file)

    data_objs = []
    data_objs = [{"text": value, "metadata": key} for key, value in data.items()]
    total = len(data_objs)
    class_obj = {
        "class": class_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "metadata",
                "dataType": ["text"],
            },
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text",
            },
        },
    }

    try:
        client.schema.create_class(class_obj)
    except:
        print("exists")
    print("------*------**")
    i = 0
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for data_obj in data_objs:
            i += 1
            if i > -1:
                loading_status = "Uploaded: " + str(i) + "/" + str(total)
                print(loading_status)
                progress_log(loading_status)

                batch.add_data_object(data_obj, class_name)
            else:
                print(i)

    res = (
        client.query.get(class_name, ["text", "metadata"])
        .with_near_text({"concepts": "What do I do with my life??"})
        .with_limit(10)
        .do()
    )
    print(res)


def delete_files(directory):
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            if not file_name.startswith("file_info"):
                file_path = os.path.join(directory, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"Deleted {file_name}")
                except Exception as e:
                    app.logger.error(f"Failed to delete {file_name}. Reason: {e}")
        print("Non-essential files deleted after vectorization.")


from datetime import datetime


def create_blank_json_with_timestamp(filename):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    name, extension = filename.rsplit(".", 1)
    filename_with_timestamp = f"{name}_{timestamp}.{extension}"

    with open(filename_with_timestamp, "w") as file:
        json.dump({}, file)

    return filename_with_timestamp


def tag_for_each_sentence(data, json_file):
    i = 0
    for name, text in data.items():
        loading_status = "pre-tagging chunk {}.".format(name)
        progress_log(loading_status)
        resp = ask_gpt_tags(text)
        resp = [tok.lower().strip() for tok in resp]
        try:
            add_key_value_to_json(json_file, text, resp)
        except Exception as e:
            print(e)
        i += 1


def add_key_value_to_json(file_path, key, value):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}
    data[key] = value
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def tag_for_each_sentence_after_delete(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump({}, json_file, indent=4)

    i = 0
    for name, text in data.items():
        loading_status = "pre-tagging chunk {}.".format(name)
        progress_log(loading_status)
        resp = ask_gpt_tags(text)
        resp = [tok.lower().strip() for tok in resp]
        try:
            add_key_value_to_json(file_path, text, resp)
        except Exception as e:
            print(e)
        i += 1


def add_key_value_to_json_after_delete(file_path, key, value):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}
    data[key] = value
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_unique_words_case_insensitive(word_dict):
    unique_words = set()  # Initialize an empty set to hold unique words

    for (
        words
    ) in (
        word_dict.values()
    ):  # Iterate through the list of words in each dictionary value
        lower_case_words = [
            word.lower() for word in words
        ]  # Convert all words to lowercase
        unique_words.update(
            lower_case_words
        )  # Add lowercase words to the set, duplicates will be ignored

    return list(unique_words)


def delete_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        print(f"The file {filename} does not exist, so it cannot be deleted.")
    else:
        print(f"The file {filename} has been successfully deleted.")


def flip_dict_and_update_as_json(input_dict, output_file):
    flipped_dict = {value: key for key, value in input_dict.items()}

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(flipped_dict)

    with open(output_file, "w") as file:
        json.dump(existing_data, file, indent=4)


def flip_dict_and_store_as_json(input_dict, output_file):
    flipped_dict = {value: key for key, value in input_dict.items()}

    with open(output_file, "w") as file:
        json.dump(flipped_dict, file, indent=4)


def flip_dict_value_and_key(input_dict):
    flipped_dict = {value: key for key, value in input_dict.items()}

    return flipped_dict


def save_json_to_directory(data, directory, filename):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")
        return

    file_path = os.path.join(directory, filename)

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error saving data to {file_path}: {e}")


@app.route("/update_files", methods=["POST"])
@admin_required
@user_activity_tracker
def update_files():
    global loading_status

    data = request.form
    layer = request.args.get("layer")

    print(layer, "---------------->>>>>>>>>>>>>>>>>SDSFDF")
    email = manage_json("read", "instanceRunning")
    print(email, "email sssssss")
    loading_status = "Update Process started..."


    if layer == "layer1":
        update_layer(data, "layer1", email)
    elif layer == "layer2":
        update_layer(data, "layer2", email)

    return render_template("chat.html")


def get_usernames():
    with open("all_users_config.json", "r") as file:
        users = json.load(file)
    usernames = [user["username"] for user in users if "username" in user]
    print("username", usernames)
    return usernames


import json


def update_configu_and_progress_from_user(user_config):
    config_path = "configu.json"
    process_path = "progress_log.json"
    try:

        new_config = {
            "layer1": {
                "openaiKey": user_config["layer1_config"]["openaiKey"],
                "layer1URL": user_config["layer1_config"]["layer1URL"],
                "layer1DriveURL": user_config["layer1_config"]["layer1DriveURL"],
                "layer1AuthKey": user_config["layer1_config"]["layer1AuthKey"],
                "classNamelayer1": user_config["layer1_config"]["classNamelayer1"],
                "process_image": user_config["layer1_config"]["process_image"],
                "layer1SliderValue": user_config["layer1_config"]["layer1SliderValue"],
            },
            "layer2": {
                "openaiKey": user_config["layer2_config"]["openaiKey"],
                "layer2URL": user_config["layer2_config"]["layer2URL"],
                "layer2DriveURL": user_config["layer2_config"]["layer2DriveURL"],
                "layer2AuthKey": user_config["layer2_config"]["layer2AuthKey"],
                "classNamelayer2": user_config["layer2_config"]["classNamelayer2"],
                "process_image": user_config["layer2_config"]["process_image"],
                "layer2SliderValue": user_config["layer2_config"]["layer2SliderValue"],
            },
        }

        new_config["openaiKey"] = user_config["openaiKey"]

        log_file = {
            "l1": user_config["l1_text"],
            "l2": user_config["l2_text"],
            "p1": user_config["p1"],
            "p2": user_config["p2"],
            "intro": user_config["intro"],
            "name": user_config["projectName"],
            "searchprompt": user_config["level2_search_prompt"],
        }

        process_log_data = read_json_file(process_path)

        save_json_to_directory(new_config, "./", config_path)

        new_log_data = {**process_log_data, **log_file}

        save_json_to_directory(new_log_data, "./", process_path)

    except KeyError as e:
        print(f"Key error: {str(e)} - Check the structure of the user configuration.")
    except FileNotFoundError:
        print(f"{config_path} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


@app.route("/set_user_context", methods=["POST"])
@login_required
def set_user_context():
    data = request.get_json()
    selected_username = data.get("username")
    print(f"Received username: {selected_username}")  # Debug print

    if not selected_username:
        return jsonify({"error": "Username not provided"}), 400

    email = get_email_from_username(selected_username)
    if email:
        session["current_email"] = email  # Store the email in the session
        manage_json("write", "instanceRunning", email)
        print(f"Session updated with email: {session['current_email']}")  # Debug output

    try:
        with open("all_users_config.json", "r") as file:
            users = json.load(file)
        user_data = next(
            (item for item in users if item["email"] == selected_username), None
        )

        print(user_data, "----------- user data form configu ")

        if user_data:
            # Update configu.json with the fetched user data
            update_configu_and_progress_from_user(user_data)
            initiate_clients()
            return jsonify(user_data), 200
        else:
            return jsonify({"error": "User not found"}), 404
    except FileNotFoundError:
        return jsonify({"error": "Configuration file not found"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_email_from_username(username):
    try:
        with open("all_users_config.json", "r") as file:
            users = json.load(file)
            user = next((u for u in users if u["username"] == username), None)
            return user["email"] if user else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_user_data():
    # Fetch the email from session
    email = manage_json("read", "instanceRunning")
    if not email:
        raise ValueError("User email not set in session")

    data_file_path = f"tag_for_each_sentence_{email}.json"
    try:
        with open(data_file_path, "r") as file:
            data = json.load(file)
            # Process the data as needed
    except FileNotFoundError:
        print(f"No such file: {data_file_path}")
        # Handle the absence of the file appropriately
    except json.JSONDecodeError:
        print("Error decoding JSON from the file")


# def get_user_id_from_username(username):
#     filename = "all_users_config.json"
#     try:
#         with open(filename, "r") as file:
#             users = json.load(file)
#             for user in users:
#                 if user.get("userName", "") == username:
#                     return user.get("userID")
#     except FileNotFoundError:
#         print(f"{filename} not found.")
#         return None
#     except json.JSONDecodeError:
#         print("Error decoding JSON.")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


def vectorize_tags(texts, class_name):
    existing_data = readAndWriteJsonData("configu.json", "r")

    layer_1 = existing_data["layer1"]

    openaiKey = existing_data["openaiKey"]

    client = weaviate.Client(
        url=layer_1["layer1URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=layer_1["layer1AuthKey"]),
        additional_headers={"X-OpenAI-Api-Key": openaiKey},
    )

    data_objs = [{"text": tx, "metadata": met} for tx, met in zip(texts, texts)]
    total = len(data_objs)

    i = 0

    class_obj = {
        "class": class_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "metadata",
                "dataType": ["text"],
            },
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text",
            },
        },
    }
    try:
        client.schema.create_class(class_obj)
    except Exception as e:
        print("Error:", e)
        print("--------*--------**")

    client.batch.configure(batch_size=50)
    with client.batch as batch:
        print("Batch=========================")
        for data_obj in data_objs:
            try:
                i += 1
                print(i)
                if i > -1:
                    loading_status = "Uploaded: " + str(i) + "/" + str(total)
                    batch.add_data_object(data_obj, class_name)
                else:
                    print("Already present.", i)
            except Exception as e:
                print(e)
                continue


    res = (
        client.query.get(class_name, ["text", "metadata"])
        .with_near_text({"concepts": "generative ai"})
        .with_limit(10)
        .do()
    )
    print(res)


def update_layer(data, layer_name, email):
    config_data = {"layer1": {}, "layer2": {}}
    config_file_path = "configu.json"

    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

    print(config_data, "config_data is here eeeeeeeeeeeeeeeee")

    manage_json(
        "write",
        "vectorFileChange",
        {
            "userName": session["email"],
            "time": datetime.now().isoformat(),
            "layer": layer_name,
        },
    )
    checkbox_status = data.get("myCheckbox", "off")
    slider_value_key = f"{layer_name.lower()}Slider"
    slider_value = data.get(slider_value_key, "0")
    process_image = True if checkbox_status == "on" else False

    layer_url_key = f"{layer_name}URL"
    layer_drive_url_key = f"{layer_name}DriveURL"
    layer_auth_key = f"{layer_name}AuthKey"
    layer_class_name_key = f"className{layer_name}"

    list_of_params = [
        layer_url_key,
        layer_drive_url_key,
        layer_auth_key,
        layer_class_name_key,
    ]

    for key in list_of_params:
        if key not in data or not data[key]:
            return jsonify({"error": "Missing values"}), 500

    layer_url = data.get(layer_url_key, "")
    layer_drive_url = data.get(layer_drive_url_key, "")
    layer_auth = data.get(layer_auth_key, "")
    layer_class_name = data.get(layer_class_name_key, "")

    existing_data = readAndWriteJsonData("configu.json", "r")

    openaiKey = existing_data["openaiKey"]

    new_data = {
        layer_url_key: layer_url,
        layer_drive_url_key: layer_drive_url,
        layer_auth_key: layer_auth,
        layer_class_name_key: layer_class_name,
        "process_image": process_image,
        f"{layer_name}SliderValue": int(slider_value),
        "openaiKey": existing_data["openaiKey"],
    }

    if existing_data:
        existing_data[layer_name] = new_data
    else:
        existing_data = {layer_name: new_data}

    readAndWriteJsonData("configu.json", "w", existing_data)

    vector_file_change_info = {
        "userName": session["email"],
        "time": datetime.now().isoformat(),
        "layer": layer_name,
    }

    progress_log_data = read_json_file("progress_log.json")

    user_config = {
        "username": email,
        "language": get_language(session.get("language", "en")),
        "projectName": progress_log_data["name"],
        "intro": progress_log_data["intro"],
        "level2_search_prompt": progress_log_data["searchprompt"],
        "l1_text": progress_log_data["l1"],
        "l2_text": progress_log_data["l2"],
        "p1": progress_log_data["p1"],
        "p2": progress_log_data["p2"],
        "layer1_config": existing_data["layer1"],
        "layer2_config": existing_data["layer2"],
        "vectorFileChange": vector_file_change_info,
        "openaiKey": openaiKey,
        "email":email
    }

    save_user_config(email, user_config)

    print(openaiKey, "---------- Open ai key from config.json")

    download_directory = ""

    if layer_name == "layer1":
        download_directory = f"Layer_1_file_{email}"
    elif layer_name == "layer2":
        download_directory = f"Layer_2_file_{email}"

    delete_pdfs_and_docxs(download_directory)

    response_data = download_folder(
        layer_drive_url, save_directory=download_directory, json_file="file_info_1.json"
    )

    if response_data.get("status") != "success":
        return jsonify({"error": "Failed to download files"}), 500

    layer_config = {
        "className": layer_class_name,
        "layerURL": layer_url,
        "layerAuthKey": layer_auth,
        "openaiKey": openaiKey,
        "folderURL": layer_drive_url,
    }
    

    tags_file_name = f"tag_for_each_sentence_{email}.json"

    if process_image == True:
        delete_file(f"flipped_dict_{email}.json")
        save_json_to_directory({},'./',tags_file_name)

        img2txt_file_name = f"image2txt_{email}.json"
        smart_chunk_json_name = f"smart_chunk_{email}.json"

        img2txt_file = create_blank_json_with_timestamp(img2txt_file_name)
        print("its working  1111 tilll here ")
        smart_chunk_json = create_blank_json_with_timestamp(smart_chunk_json_name)
        print("its working till  2222 hereeee...........")
        loading_status = "processing files."
        progress_log(loading_status)
        print("its working till  2233322 hereeee...........")
        process_all_pdfs(download_directory, openaiKey, img2txt_file)
        loading_status = "Smart chunking...."
        progress_log(loading_status)
        print("its working till  000000 hereeee...........")
        smart_chunk(img2txt_file, smart_chunk_json)
        smart_chunk_data = read_json_file(smart_chunk_json)

        flip_dict_and_store_as_json(smart_chunk_data, f"flipped_dict_{email}.json")
        wrdata = read_json_file(f"flipped_dict_{email}.json")

        flip_dict_and_store_as_json(wrdata, f"allsentences_{email}.json")
        loading_status = "pre-tagging chunks."
        progress_log(loading_status)
        print("its working till 3 hereeeeeeeeeeeeeeeeeeee")
        tag_for_each_sentence(smart_chunk_data, tags_file_name)
        print("its working till working 4444")
        tags = read_json_file(tags_file_name)
        tags = get_unique_words_case_insensitive(tags)
        vectorize_tags(tags, layer_class_name)
        print(layer_class_name, "layer of class name ssssssss")
    else:
        pdf_vectorization(download_directory, layer_config, set(), int(slider_value))

    initiate_clients()
    save_user_config(email, user_config)
    print("Files downloaded and vectorized.")
    loading_status = "Ready to use..."
    progress_log(loading_status)
    copy_files(download_directory, f"{download_directory}_all_files")
    delete_files(download_directory)
    return jsonify({"success": "Done"}), 500


def copy_files(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    files = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]

    for file in files:
        if file.startswith("modi_"):
            modified_filename = file[5:]
        else:
            modified_filename = file

        src_file_path = os.path.join(source_dir, file)
        dest_file_path = os.path.join(dest_dir, modified_filename)

        if os.path.exists(dest_file_path):
            src_mod_time = os.path.getmtime(src_file_path)
            dest_mod_time = os.path.getmtime(dest_file_path)

            if src_mod_time > dest_mod_time:
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Updated {modified_filename} in {dest_dir}")
            else:
                print(f"No update needed for {modified_filename} in {dest_dir}")
        else:

            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied {modified_filename} to {dest_dir}")


@app.route("/admin_list", methods=["GET", "POST", "PATCH", "DELETE"])
@admin_required
@user_activity_tracker
def admin_list():
    try:
        if request.method == "GET":
            return render_template("admin.html")

        elif request.method == "PATCH":
            # Parse request data
            data = request.json
            user_id = data.get("userId")
            role = data.get("role")

            # Update user's admin status in BigQuery
            update_admin_status(user_id, role)

            return jsonify({"message": "Admin status updated successfully"})

        elif request.method == "DELETE":
            # Parse request data
            user_id = request.args.get("userId")
            # Delete user from BigQuery
            status = delete_user("jobbot-415816.jobBot.users", user_id)
            if status:
                return jsonify({"message": "User deleted successfully"})
            else:
                return jsonify(
                    {"message": "User deleted successfully", "status": status}
                )
    except Exception as e:

        return jsonify({"error": str(e)})


def update_admin_status(user_id, role):
    try:
        # Initialize BigQuery client
        table_id = "jobbot-415816.jobBot.users"
        # Define the update query
        update_query = f"""
            UPDATE `{table_id}`
            SET role = '{role}'
            WHERE userID = '{user_id}'
        """

        # Build the job configuration
        job_config = bigquery.QueryJobConfig()

        # Run the update query
        query_job = dbclient.query(update_query, job_config=job_config)

        # Wait for the query to complete
        query_job.result()

        print(f"Role updated to admin for user ID: {user_id}")

    except Exception as e:
        print(f"Error updating admin status for user ID {user_id}: {e}")


def delete_user(tableId, user_id):
    try:
        # Define the delete query
        query = f"""
        DELETE FROM `{tableId}`
        WHERE userID = '{user_id}'
        """

        # Execute the delete query
        query_job = dbclient.query(query)
        query_job.result()  # Wait for the query to complete

        print(f"User with userID {user_id} deleted successfully")

    except Exception as e:
        print(f"Error deleting user with user ID {user_id}: {e}")


@app.route("/whiteList", methods=["POST", "DELETE"])
@admin_required
@user_activity_tracker
def update_user_WhiteList():
    if request.method == "POST":
        try:
            dataset_name = "jobbot-415816.jobBot.whitelisted_users"
            table_id = f"{dataset_name}"

            data = request.json
            email = data["email"]
            user_id = str(uuid.uuid4())

            query = f"""
                INSERT INTO `{table_id}` (email, userID)
                VALUES (@email, @userID)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("email", "STRING", email),
                    bigquery.ScalarQueryParameter("userID", "STRING", user_id),
                ]
            )

            query_job = dbclient.query(query, job_config=job_config)
            query_job.result()

            return {"message": "User WhiteList updated successfully"}
        except Exception as e:
            return {"message": f"Error updating user WhiteList: {str(e)}"}, 500
    elif request.method == "DELETE":
        try:
            data = request.json
            user_id = data["userID"]
            if not user_id:
                return {"message": "User ID is required for deletion"}, 400

            delete_user("jobbot-415816.jobBot.whitelisted_users", user_id)

            return {"message": f"User with ID {user_id} deleted successfully"}
        except Exception as e:
            return {"message": f"Error deleting user: {str(e)}"}, 500


def makeThemWhiteList():
    try:
        table_id = "jobbot-415816.jobBot.users"

        query = f"""
            UPDATE `{table_id}`
            SET iswhiteList = FALSE
            WHERE role = 'user'
        """

        job_config = bigquery.QueryJobConfig()

        query_job = dbclient.query(query, job_config=job_config)
        results = query_job.result()
        print("Update successful")
    except Exception as e:
        print(f"Error updating user whitelist: {str(e)}")


def get_whitelisted_users(email=None):
    try:
        table_id = "jobbot-415816.jobBot.whitelisted_users"

        if email:
            sql_query = f"""
                SELECT * 
                FROM `{table_id}`
                WHERE email = '{email}'
            """
        else:
            sql_query = f"""
                SELECT * 
                FROM `{table_id}`
            """

        query_job = dbclient.query(sql_query)
        results = query_job.result()

        user_data = [dict(row) for row in results]
        return user_data
    except Exception as e:
        return ({"error": str(e)}), 500


@app.route("/whiteListUsers")
@admin_required
@user_activity_tracker
def whiteListUsers():
    return jsonify(get_whitelisted_users())


@app.route("/check_for_whitelList", methods=["POST"])
@admin_required
@user_activity_tracker
def update_whitelist_status():
    data = request.json
    status = data.get("status", False)
    manage_json("write", "whitelist", value=status)
    read_json_file
    return jsonify({"message": "WhiteList status updated successfully"}), 200


@app.route("/getLocations", methods=["GET", "POST"])
@login_required
@user_activity_tracker
def get_locations():
    table_id = "jobbot-415816.jobBot.locations"

    if request.method == "POST":
        try:
            # Get longitude and latitude from the request
            data = request.json
            longitude = data.get("longitude")
            latitude = data.get("latitude")

            locationID = str(uuid.uuid4())
            rows_to_insert = [
                {"locationID": locationID, "longitude": longitude, "latitude": latitude}
            ]
            errors = dbclient.insert_rows_json(
                table_id, rows_to_insert
            )  # Wait for the query to complete

            if errors == []:
                return jsonify({"message": "Location added successfully"}), 200
            else:
                return jsonify({"error": "Failed to add location"}), 500

        except Exception as e:
            # Handle any errors and return an error message
            return jsonify({"error": str(e)}), 500

    elif request.method == "GET":
        try:
            # Retrieve all locations from the BigQuery table
            query = f"""
                SELECT *
                FROM `{table_id}`
            """
            query_job = dbclient.query(query)
            results = query_job.result()

            locations = [dict(row) for row in results]

            return jsonify(locations)

        except Exception as e:
            # Handle any errors and return an error message
            return jsonify({"error": str(e)}), 500


##response by layer wise


@app.route("/changeLayer", methods=["POST"])
@admin_required
@user_activity_tracker
def change_layer():
    global layerLevel
    selected_level = request.form["selected_level"]
    print("Selected level:", selected_level)
    layerLevel = selected_level
    manage_json("write", "layerLevel", value=layerLevel)
    return jsonify({"level": layerLevel})


@app.route("/map")
@admin_required
@user_activity_tracker
def showmap():
    return render_template("map.html")


@app.route("/logout", methods=["GET"])
@login_required
def logout():
    try:
        if session["role"] == "admin":
            updateAdminFlag(False)
        print("\n\n\n\n\n\nSession LOGOUT\n\n\n\n\n")
        session.clear()
        return jsonify({"message": "User logout successfully"})

    except Exception as e:
        return jsonify({"message": str(e)})


custom_function_gt = [
    {
        "name": "return_match",
        "description": "to be used to return the match boolean",
        "parameters": {
            "type": "object",
            "properties": {
                "match": {
                    "type": "boolean",
                    "description": "return True if the two responses match with an overlap of over 70%. Elsem return False",
                },
            },
            "required": ["match"],
        },
    }
]


def checkGroundTruth(question, answer, groundTruth):
    system_message = "You are an Analyzer. You will be given a question and its answer along with the Ground Truth. Your task is to compare both the answer and the ground truth. If there is an overlap of over 70% between the answer and the actual ground truth, return True. Else, return False. Also, sometimes, the answer might contain additional details not present in the ground truth. As long as 70% or more of the ground truth is present in the answer, you must return True."

    user_message = (
        f"Question:<{question}>Answer:[{answer}]GroundTruth:{{{groundTruth}}}"
    )

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        max_tokens=1000,
        temperature=0.0,
        functions=custom_function_gt,
        function_call="auto",
    )
    print(response, "compartion of ground truth vs model ans")
    reply = response.choices[0].message.content
    try:
        reply = json.loads(response.choices[0].message.function_call.arguments)["match"]
    except Exception as e:
        print(e)
        reply = "error in analysis"

    return reply


@app.route("/getExcelSheet", methods=["POST"])
def getExcelSheet():
    data = request.get_json()

    total_questions = 0
    correct_answers = 0
    for ele in data:
        boolean = checkGroundTruth(ele["question"], ele["answer"], ele["groundTruth"])
        ele["boolean"] = f"{boolean}".capitalize()
        total_questions += 1
        if boolean == True:
            correct_answers += 1

    if total_questions > 0:
        effectiveness = (correct_answers / total_questions) * 100
    else:
        effectiveness = 0

    df = pd.DataFrame(data)

    effectiveness_row = pd.DataFrame(
        {
            "question": ["Effectiveness"],
            "answer": [""],
            "groundTruth": [""],
            "boolean": [f"{effectiveness:.2f}%"],
        }
    )
    df = pd.concat([df, effectiveness_row], ignore_index=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Questions and Answers")
        workbook = writer.book
        worksheet = writer.sheets["Questions and Answers"]
        max_row = worksheet.max_row
        bold_font = Font(bold=True)
        worksheet["A" + str(max_row)].font = bold_font
        worksheet["B" + str(max_row)].font = bold_font

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="questions_answers.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/update_apikey", methods=["POST"])
def update_openAIKey():

    if request.is_json:
        form_data = request.get_json()
        openaiKey = form_data.get("openaiKey", "")
    else:
        return jsonify({"error": "Invalid data format"}), 400

    try:
        existing_data = readAndWriteJsonData("configu.json", "r")

        existing_data["openaiKey"] = openaiKey
        existing_data["layer1"]["openaiKey"] = openaiKey
        existing_data["layer2"]["openaiKey"] = openaiKey
        readAndWriteJsonData("configu.json", "w", existing_data)
        initiate_clients()
        return jsonify({"message": "OpenAI Key updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


initiate_clients()
import os
import json


def save_user_config(email, new_config):
    filename = "all_users_config.json"
    data = []
    new_data = []

    print(email, new_config, "email and new config ------->>>>>>")

    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, "r") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print("JSON decode error, starting with an empty list.")
        except Exception as e:
            print(f"Error reading from file: {e}")
            return False

    # Check if the configuration for the specified email already exists
    updated = False
    for entry in data:
        if entry["email"] == email:
            # Update the existing configuration
            entry.update(new_config)
            print(f"Updated configuration for user {email}.")
            updated = True
            break

    # If the configuration for the email wasn't found, add a new entry
    if not updated:
        new_config["email"] = email  # Ensure the email is part of the configuration
        data.append(new_config)
        print(f"Added new configuration for user {email}.")

    # Write the updated data back to the file

    try:
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print("Configuration saved successfully.")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False


import fnmatch


def delete_unwanted_files(directory):
    patterns = ["image2txt_*", "smart_chunk_*"]
    for root, dirs, files in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    # print(f'Deleted: {file_path}')
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")


repo_directory = "/"
delete_unwanted_files(repo_directory)

import os


def ensure_directories_exist(dir1, dir2):
    for directory in [dir1, dir2]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        else:
            print(f"Directory already exists: {directory}")


if __name__ == "__main__":
    app.run(debug=True)
