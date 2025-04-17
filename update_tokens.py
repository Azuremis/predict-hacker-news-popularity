import json

# Read the original JSON file
with open('tokens/tokens.json', 'r') as f:
    data = json.load(f)

# Increment each value by 1
updated_data = {k: v + 1 for k, v in data.items()}

# Write the updated JSON back to the file
with open('tokens/tokens.json', 'w') as f:
    json.dump(updated_data, f, indent=2)

print("Successfully updated all token values by adding 1") 