import json


def save_text(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(file_path, data, indent=4):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
