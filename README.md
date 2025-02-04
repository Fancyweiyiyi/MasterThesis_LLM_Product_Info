
# Master Thesis: Discovery and Normalization of Long-Tail Product Attributes Using Large Language Models (LLMs)

This repository contains the materials and code developed for the master thesis titled **"Discovery and Normalization of Long-Tail Product Attributes Using LLMs"**. The project focuses on evaluating the performance of state-of-the-art large language models (LLMs) for extracting and normalizing product attributes from unstructured textual data in e-commerce scenarios.

## Repository Structure

The repository is organized into the following folders:

### 1. **Data**
   - **`Data/wdc_products/`**: Contains the initial dataset provided by Alexander Brinkmann[2], sourced from the WDC Product Attribute-Value Extraction benchmark.
   - The dataset serves as the starting point for the project, including raw product titles and descriptions.

### 2. **Dataset**
   - **Final Dataset**: Includes processed data for four product categories: **Grocery, Home, Jewelry, and Office**.
   - Available in two formats:
     - **CSV**: For tabular data representation.
     - **JSON**: For hierarchical data representation.

### 3. **Model_Output**
   - Contains the outputs generated by the evaluated models:
     - **GPT-4**
     - **GPT-4o Mini**
     - **LLaMA 3.1**
     - **DeepSeek-V3**
   - Each output folder corresponds to a specific model and includes extracted attributes for the four categories.

### 4. **Prompt Design**
   - Documents the iterative process of prompt engineering to optimize the performance of LLMs for attribute extraction.
   - Contains examples of prompt designs and their evolution through testing and refinement.

### 5. **Evaluation**
   - This folder contains subfolders for each evaluation stage:
     - **`Basic Evaluation/`**: Evaluates the raw outputs of the models based on metrics like Precision, Recall, F1-Score, and Coverage.
     - **`manual_map_evaluation/`**: Uses manually curated mappings to align extracted attributes with ground truth data.
     - **`CF1_Evaluation/`**: Implements the Classification Macro F1 (CF1) metric to assess performance across frequent and long-tail attributes.
     - **`Advanced_evaluation/`**: Applies clustering algorithms (K-Means, DBSCAN, Agglomerative) and evaluates semantic groupings using metrics like Silhouette Score and Davies-Bouldin Index.
     - **`Truth_dataset/`**: Includes the ground truth dataset used for comparison and benchmarking.

---

## Research Objectives

1. **Performance Evaluation**:
   - Assess the ability of GPT-4, GPT-4o Mini, LLaMA 3.1, and DeepSeek-V3 to extract and normalize product attributes.
   - Focus on metrics like Precision, Recall, F1-Score, Coverage, and Classification Macro F1 (CF1).

2. **Model Comparison**:
   - Highlight the strengths, limitations, and overall suitability of each model for e-commerce attribute extraction.

3. **Advanced Evaluation**:
   - Analyze the semantic relationships of attributes using clustering algorithms and embedding-based metrics.

---



## Key Findings

### GPT-4
- **Strengths**:
  - Achieved the **highest Recall (0.59)** and **Coverage (15.05%)** in the basic evaluation, showcasing its ability to extract a wide range of attributes.
  - Performed well in extracting **frequent attributes**, especially after manual mapping, with near-perfect precision in categories like **Home** and **Jewelry**.
  - Formed **semantically meaningful clusters** in categories like **Grocery**, particularly for attributes related to packaging and certifications.
- **Weaknesses**:
  - Struggled with **overgeneration**, producing irrelevant or redundant attributes, leading to low Precision (0.05) in the basic evaluation.
  - Faced challenges in extracting **long-tail attributes**, where variability and semantic complexity impacted performance.
  - Redundant clustering results affected interpretability.
- **Insights**:
  - GPT-4 is suitable for tasks prioritizing **breadth** and **coverage** but requires post-processing to improve precision and reduce noise.

---

### GPT-4o Mini
- **Strengths**:
  - Demonstrated **higher Precision (0.18)** in the basic evaluation, excelling in filtering irrelevant outputs.
  - Produced **compact and precise clusters**, particularly in **Jewelry**, extracting attributes like **stone cut** and **metal purity** accurately.
  - Showed balanced performance for frequent attributes in manual mapping evaluations, with significant improvements in F1-scores.
- **Weaknesses**:
  - Limited in **Recall (0.15)** and **Coverage (0.82%)**, resulting in difficulty capturing diverse or long-tail attributes.
  - Missed critical long-tail attributes in categories like **Grocery** (e.g., "BPA free") and **Office**.
  - Clustering often lacked **diversity**, grouping redundant attributes together.
- **Insights**:
  - GPT-4o Mini is ideal for applications requiring **high precision** and **compact outputs**, but it needs improvements to handle **attribute diversity**.

---

### LLaMA 3.1
- **Strengths**:
  - Performed well in extracting **frequent attributes** after manual mapping, achieving **high F1-scores** in categories like **Home** and **Jewelry**.
  - Demonstrated **contextual relevance**, particularly for attributes like **design patterns** and **material type** in **Home**.
  - Formed **well-separated clusters** in specific domains after manual refinement.
- **Weaknesses**:
  - Delivered **near-zero metrics** in the basic evaluation, failing to extract both frequent and long-tail attributes effectively.
  - Required significant preprocessing and manual mapping for alignment with ground truth attributes.
  - Struggled with **semantic inconsistencies**, grouping attributes like "Net Weight" and "Gross Weight" together despite distinct meanings.
- **Insights**:
  - LLaMA 3.1 benefits most from **manual refinement**, making it a viable option for exploratory tasks requiring additional intervention.

---

### DeepSeek-V3
- **Strengths**:
  - Demonstrated a **balanced performance** across frequent and long-tail attributes, achieving a **Macro F1 score of 0.59**.
  - Excelled in categories like **Home**, extracting attributes like **material quality** and **energy efficiency** effectively.
  - Showed robustness in semantic grouping, with a relatively low **Davies-Bouldin Index (1.23)** compared to other models.
- **Weaknesses**:
  - Moderate Precision (0.09 in the basic evaluation), indicating occasional irrelevant or redundant outputs.
  - Faced challenges with **long-tail attributes** in categories like **Jewelry** and **Office**, where semantic complexity was high.
- **Insights**:
  - DeepSeek-V3 offers a middle ground between **diversity** and **compactness**, making it suitable for applications requiring balanced performance in attribute extraction.

---

These detailed insights provide a nuanced understanding of each model's capabilities, guiding researchers and practitioners in selecting the appropriate model for specific e-commerce attribute extraction tasks.

---

## How to Use

1. **Data Preparation**:
   - Start with the raw dataset in `Data/wdc_products/`.
   - Use the processed datasets in the `Dataset/` folder for experiments.

2. **Model Execution**:
   - Run the models using the designed prompts available in the `Prompt Design/` folder.
   - Outputs can be found in `Model_Output/`.

3. **Evaluation**:
   - Evaluate model performance using scripts in the `Evaluation/` folder.
   - Start with `Basic Evaluation/` and progress through manual mapping and advanced clustering techniques.

---

## Future Directions

- **Improved Embeddings**: Explore domain-specific embeddings to enhance semantic alignment.
- **Hybrid Clustering**: Combine Levenshtein-based and embedding-based clustering for better coherence.
- **Automated Mapping**: Develop automated techniques to reduce reliance on manual efforts.

---

## Citation

If you use this repository, please cite:

> **Wei Yiyi**, "Discovery and Normalization of Long-Tail Product Attributes Using LLMs", Master Thesis, 2025.

---

## Contact

For questions or collaborations, please contact **Wei Yiyi** at [yiyiwei2000@gmail.com].

## References
[1] Nick Baumann, Alexander Brinkmann, and Christian Bizer. Using llms for the extraction
and normalization of product attribute values. arXiv preprint arXiv:2403.02130, 2024.

[2] Alexander Brinkmann, Roee Shraga, and Christian Bizer. Product attribute value extraction
using large language models. arXiv preprint arXiv:2310.12537, 2023.

[3] Alexander Brinkmann, Roee Shraga, Reng Chiz Der, and Christian Bizer. Product informa-
tion extraction using chatgpt. arXiv preprint arXiv:2306.14921, 2023
