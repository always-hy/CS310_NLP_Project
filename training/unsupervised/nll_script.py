from run_nll import compute_nll_from_json  # if in another file

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="human",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_human.txt",
#     model="gpt2"
# )

compute_nll_from_json(
    json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
    key="gpt",  # or "gpt"
    model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
    output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_gpt.txt",
    model="gpt2"
)

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="claude",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_claude.txt",
#     model="gpt2"
# )

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="gpt_prompt1",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_gpt_prompt1.txt",
#     model="gpt2"
# )

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="gpt_prompt2",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_gpt_prompt2.txt",
#     model="gpt2"
# )

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="gpt_semantic",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_gpt_semantic.txt",
#     model="gpt2"
# )

# compute_nll_from_json(
#     json_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/dataset/pairwaise_english/essay_data.json",
#     key="gpt_writing",  # or "gpt"
#     model_path="D:/IMPORTANT/final_semester/modelscope_cache/models/openai-community/gpt2",
#     output_nll_file="D:/IMPORTANT/final_semester/CS310_NLP_Project/results/nll_output_essay_gpt_writing.txt",
#     model="gpt2"
# )