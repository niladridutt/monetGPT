import json
import random
import os


def load_jsons(json_paths):
    """
    Given a list of JSON file paths, load each JSON.
    Assumes each JSON is either a list of dicts or a single dict.
    If a JSON is a list, its items will be added to the combined list.
    If a JSON is a dict, it will be appended as one entry.
    """
    combined = []
    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined.extend(data)
                elif isinstance(data, dict):
                    combined.append(data)
                else:
                    print(f"Warning: {path} does not contain a list or dict. Skipping.")
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return combined

def main():
    json_paths = [
        'data/sharegpt_puzzle_1.json',
        'data/json/sharegpt_puzzle_2json',
        'data/json/sharegpt_puzzle_3.json'
    ]

    # Load and combine JSON entries
    combined_entries = load_jsons(json_paths)

    # Shuffle the outer list of entries (each entry remains intact)
    random.shuffle(combined_entries)

    # Ensure the output directory exists
    output_path = './train/data/monetgpt.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the shuffled combined entries to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_entries, f, ensure_ascii=False, indent=4)
        print(f"Combined and shuffled JSON saved to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")

if __name__ == '__main__':
    main()
