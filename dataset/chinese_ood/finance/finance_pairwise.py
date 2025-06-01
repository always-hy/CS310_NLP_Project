import json
import uuid

input_file = "finance.jsonl"
output_data = []

try:
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            human_answers = entry.get("human_answers", [])
            chatgpt_answers = entry.get("chatgpt_answers", [])
            min_length = min(len(human_answers), len(chatgpt_answers))
            for i in range(min_length):
                human_text = human_answers[i]
                chatgpt_text = chatgpt_answers[i]
                if human_text and chatgpt_text:
                    # Clean up whitespace
                    human_text = " ".join(human_text.split())
                    chatgpt_text = " ".join(chatgpt_text.split())
                    output_data.append({"Human": human_text, "Generated": chatgpt_text})

except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: File '{input_file}' contains invalid JSON.")
    exit(1)

output_file = "finance_pairwise.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Dataset converted and saved to '{output_file}'")
except Exception as e:
    print(f"Error writing to '{output_file}': {e}")
