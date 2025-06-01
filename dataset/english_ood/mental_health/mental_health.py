import json

# Load the input JSON file
input_file = "mental_health_pairwise.json"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: File '{input_file}' is not a valid JSON file.")
    exit(1)

# Convert the dataset
output_data = []
for entry in input_data:
    # Get Human and Generated responses, ignoring Context
    human_text = entry.get("Human", "")
    generated_text = entry.get("Generated", "")

    # Clean up extra whitespace
    human_text = " ".join(human_text.split())
    generated_text = " ".join(generated_text.split())

    # Add Human response with label 0
    if human_text:  # Only add if text is not empty
        output_data.append({"text": human_text, "label": 0})

    # Add Generated response with label 1
    if generated_text:  # Only add if text is not empty
        output_data.append({"text": generated_text, "label": 1})

# Save to output JSON file
output_file = "mental_health.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Dataset converted and saved to '{output_file}'")
except Exception as e:
    print(f"Error writing to '{output_file}': {e}")
