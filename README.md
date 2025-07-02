# CS310 NLP Project: Detecting Human vs. LLM-Generated Texts

This repository contains the implementation of the CS310 Natural Language Processing final project at Southern University of Science and Technology. The project focuses on detecting human-written versus large language model (LLM)-generated texts using supervised learning and zero-shot approaches across English and Chinese datasets.

## Project Overview

The project implements two approaches to distinguish human-written texts from LLM-generated ones:

1. **Supervised Learning**: Fine-tuning a BERT model (`bert-base-uncased`) on the `gb-dataset` (English) and `face-dataset` (Chinese) for text classification, with performance evaluated on in-domain and out-of-domain (OOD) datasets.
2. **Zero-shot Detection**: Using FourierGPT with pairwise heuristic-based methods and GPT-2 to compute negative log-likelihood (NLL) scores with z-score normalization, tested on `gb-pair-comp-dataset` (English) and `face-pair-dataset` (Chinese).

### Datasets

- **English**:
  - `gb-dataset`: 21,994 samples (8,000 reuter, 7,000 wp, 6,994 essay; 2,994 human, 19,000 AI-generated).
  - `gb-ood-dataset`: OOD testing with 1,800 poet and 602 mental health conversation samples.
  - `gb-pair-comp-dataset`: 302 poet and 200 mental health conversation pairs for zero-shot.
- **Chinese**:
  - `face-dataset`: 14,967 AI-generated and 13,014 human-generated texts (news, webnovel, wiki).
  - `face-pair-dataset`: 4,530 news, 4,977 webnovel, 3,607 wiki pairwise texts.
  - `face-ood-dataset`: OOD testing with finance, law, and medicine domains.

### Key Findings

- Supervised BERT models perform well on in-domain datasets but struggle with OOD performance.
- Zero-shot FourierGPT shows better OOD performance but lower in-domain accuracy compared to supervised methods.
- AI-generated texts tend to have higher average power values in selected frequency components in FourierGPT.

## Installation

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## Team Members

| Name                | Student ID | Email                        |
| ------------------- | ---------- | ---------------------------- |
| Fitria Zusni Farida | 12112351   | 12112351@mail.sustech.edu.cn |
| Sreyny Tha          | 12113055   | 12113055@mail.sustech.edu.cn |
| Tan Hao Yang        | 12212027   | 12212027@mail.sustech.edu.cn |
