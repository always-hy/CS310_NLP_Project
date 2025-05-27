import json

# Initialize a list to store the processed data
data = []

# Read the local medicine.jsonl file
with open("medicine.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line.strip())

        # Add human answers (label 0)
        for human_answer in entry["human_answers"]:
            if human_answer:  # Ensure the answer is not empty
                data.append({"text": human_answer, "label": 0})

        # Add AI answers (label 1)
        for chatgpt_answer in entry["chatgpt_answers"]:
            if chatgpt_answer:  # Ensure the answer is not empty
                data.append({"text": chatgpt_answer, "label": 1})

# Save to JSON file
with open("processed_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Local dataset processed and saved as 'processed_dataset.json'")
