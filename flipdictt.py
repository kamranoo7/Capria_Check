import json

def flip_dict_and_store_as_json(input_dict, output_file):
    flipped_dict = {value: key for key, value in input_dict.items()}

    with open(output_file, 'w') as file:
        json.dump(flipped_dict, file, indent=4)

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    
    return data

input_dict = read_json_file("tag_each_sentence_april11.json")

output_file = 'flipped_dict.json'  # Specify your desired output file name
flip_dict_and_store_as_json(input_dict, output_file)
