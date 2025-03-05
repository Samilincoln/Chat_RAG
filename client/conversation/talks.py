import os
import json
import re

def load_small_talks():
    """Loads small talk responses from a JSON file located in the same directory as app.py."""
    json_path = "small_talks.json"  # Direct relative path

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(json_path)}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)

small_talks = load_small_talks()

def clean_input(user_input):
    """Removes punctuation and converts input to lowercase."""
    return re.sub(r'[^\w\s]', '', user_input).strip().lower()


