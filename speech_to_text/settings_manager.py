import json
import re
import os

def load_settings_from_file(file_path):
    """Load settings from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_settings_to_file(settings, file_path):
    """Save settings to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(settings, file, indent=4)

def get_filtered_settings(settings, valid_keys):
    """Filter settings based on a list of valid keys."""
    return {key: settings[key] for key in settings if key in valid_keys}

def parse_number_list(number_list_str):
    """Parse a string of numbers separated by commas into a list of floats."""
    if ',' in number_list_str:
        return list(map(float, number_list_str.split(',')))
    return float(number_list_str)

def parse_int_list(int_list_str):
    """Parse a string of integers separated by commas into a list of integers."""
    if ',' in int_list_str:
        return list(map(int, int_list_str.split(',')))
    return [int(int_list_str)]

def filter_transcribe_settings(settings):
    """Filter and adjust transcribe settings based on expected data types."""
    valid_keys = settings.keys()  # Assuming all keys are valid for simplicity
    filtered_settings = get_filtered_settings(settings, valid_keys)

    if 'temperature' in filtered_settings and isinstance(filtered_settings['temperature'], str):
        filtered_settings['temperature'] = parse_number_list(filtered_settings['temperature'])

    if 'suppress_tokens' in filtered_settings:
        suppress_tokens = filtered_settings['suppress_tokens']
        if isinstance(suppress_tokens, str) and re.match(r'^(-?\d+|(-?\d+,)+-?\d+)$', suppress_tokens):
            filtered_settings['suppress_tokens'] = parse_int_list(suppress_tokens)
        elif isinstance(suppress_tokens, int):
            filtered_settings['suppress_tokens'] = [suppress_tokens]
        else:
            raise ValueError("suppress_tokens must be an int or comma-separated string of ints.")

    return filtered_settings

# Example of how to use these functions if settings are saved in a JSON file
# if __name__ == "__main__":
#     # Set the absolute path for the user settings file
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     settings_path = os.path.join(BASE_DIR, "user_settings.json")
#     user_settings = load_settings_from_file(settings_path)
#     print("Loaded settings:", user_settings)

#     # Example filtering for a specific module
#     transcribe_settings = filter_transcribe_settings(user_settings.get('transcribe_settings', {}))
#     print("Filtered transcribe settings:", transcribe_settings)
