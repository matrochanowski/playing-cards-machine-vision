import json

json_file = 'jsons/images.json'
with open(json_file, 'r') as file:
    data = json.load(file)

# Sprawdź główne klucze
if isinstance(data, dict):
    print("Główne klucze:", data.keys())
elif isinstance(data, list):
    print(f"JSON to lista o {len(data)} elementach")


def explore_structure(data, level=0):
    indent = "  " * level  # poziom wcięcia

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}{key}:")
            explore_structure(value, level + 1)
    elif isinstance(data, list):
        print(f"{indent}Lista o {len(data)} elementach:")
        if len(data) > 0:
            explore_structure(data[0], level + 1)  # Zajrzyj do struktury pierwszego elementu listy
    else:
        print(f"{indent}{type(data).__name__}")
