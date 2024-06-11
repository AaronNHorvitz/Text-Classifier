# TopicMiner

TopicMiner is a comprehensive text analytics toolkit designed to process and analyze email data to extract meaningful insights. It utilizes advanced natural language processing techniques, including topic modeling and text classification, to categorize and visualize email content efficiently. The toolkit is structured to support modular development and easy scalability.

## Project Structure
```

TopicMiner/
│
├── topicminer/
│	├── models/
│	│   ├── __init__.py
│	│   ├── topic_models/
│	│   │   ├── __init__.py
│	│   │   ├── word_2_vec_k_means.py
│	│   │   └── lda_model.py
│	│   └── classification_models/
│	│       ├── __init__.py
│	│       └── bert_classifier.py
│	│
│	├── visualizations/
│	│   ├── __init__.py
│	│   ├── lda_viz.py
│	│   ├── embeddings_viz.py
│	│   └── reports_viz.py
│	│
│   ├── utils/
│	│   ├── __init__.py
│	│   ├── email_etl.py
│	│   ├── text_processing.py
│	│   └── widgets.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│
├── tests/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_lda_model.py
│   │   └── test_word_2_vec_k_means.py
│   │
│   ├── visualizations/
│   │   ├── __init__.py
│   │   ├── test_lda_visualization.py
│   │   └── test_embeddings_visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── test_email_etl.py
│       ├── test_text_processing.py
│       └── test_widgets.py
│
├── data/
│   ├── raw_data/
│   ├── unwanted_texts/
│   └── processed_data/
│
├── notebooks/
│
├── sandbox/
│
├── output/
│   └── model_results.xlsx
│
├── setup.py
├── requirements.txt
└── README.md
```

### Description of Directories

- `models/`: Contains all the machine learning models used in TopicMiner, including both topic models and classification models.
- `visualizations/`: Scripts and modules dedicated to the visualization of data and model outcomes.
- `utils/`: Utility scripts including data extraction, transformation, and loading (ETL) processes.
- `tests/`: Contains unit tests for each of the components ensuring reliability and functionality.
- `data/`: Storage for raw data, processed data, and any scripts or files needed for processing unwanted text.
- `config/`: Configuration settings for the project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and interactive coding.
- `sandbox/`: Space for experimental scripts and trial code.
- `output/`: Directory for storing output files like the Excel file with model results.

### Usage

This section provides a quick guide on how to start using TopicMiner to analyze email data:

1. **Set Up Your Environment**:
   Ensure you have Python installed and then install the dependencies:
   ```bash
   pip install -r requirements.txt

2. **Prepare Your Data**:
Place your raw email data in the data/raw_data directory.

3. **Running Analysis**:
You can start with the Jupyter notebooks provided in the notebooks/ directory for guided analysis, or use the scripts in models/ and visualizations/ for more automated processes.

4. **Visualizing Data**:
Use the scripts in the visualizations/ directory to generate visual reports and insights from the analyzed data.

5. **Export Results**:
Check the output/ directory for results and reports, including the Excel file with categorized emails.

