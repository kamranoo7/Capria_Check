import os 
import json
from datetime import datetime
from dateutil import parser
# datetime_obj = parser.parse(serializable_now)

def updateAdminFlag(value):
    manage_json('write', 'adminLoginFlag', value)


def getAdminFlag():
    adminLoginFlag = manage_json("read",'adminLoginFlag')
    return adminLoginFlag



def updateAdminLastActivity(time):
    print(f"{time}",'time--------------->>>>')
    manage_json('write', 'adminLastActivity', f"{time}")


def getAdminLastActivity():
    adminLoginLastActivity = manage_json("read",'adminLastActivity')
    return parser.parse(adminLoginLastActivity)



def manage_json(action, field_name, value=None, file_path='progress_log.json'):
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print("Error reading JSON file. File could be empty or corrupt.")
            return "Error reading JSON file."

    if action == 'write':
        data[field_name] = value
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        return None  
    elif action == 'read':
        try:
            return data[field_name]
        except KeyError:
            return 0
    else:
        return "Invalid action specified."
