# TopicMiner

TopicMiner is a comprehensive text analytics toolkit designed to process and analyze email data to extract meaningful insights. It utilizes advanced natural language processing techniques, including topic modeling and text classification, to categorize and visualize email content efficiently. The toolkit is structured to support modular development and easy scalability.

## Table of Contents
- [TopicMiner](#topicminer)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Description of Directories](#description-of-directories)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Set Up Your Environment](#1-set-up-your-environment)
    - [2. Prepare Your Data](#2-prepare-your-data)
    - [3. Running Analysis](#3-running-analysis)
    - [4. Visualizing Data](#4-visualizing-data)
    - [5. Export Results](#5-export-results)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Features
- **Topic Modeling**: Utilize advanced techniques like LDA, Doc2Vec with K-Means, and Word2Vec with K-Means to uncover hidden topics in email data.
- **Text Classification**: Implement robust classification models including Random Forest and XGBoost classifiers for categorizing emails.
- **Data Visualization**: Generate insightful visualizations to interpret model outcomes and data trends.
- **Modular Design**: Easily extend and scale the toolkit with a well-structured and modular codebase.
- **Comprehensive Testing**: Ensure reliability with extensive unit tests covering all components.

## Project Structure
```
TopicMiner/
│
├── topicminer/
│	│	
│	├── __init__.py
│	│   
│	├── models/
│	│   ├── __init__.py
│	│   │ 
│	│   ├── topic_models/
│	│   │   ├── __init__.py
│	│   │   ├── doc2vec_kmeans.py
│	│   │   ├── word2vec_kmeans.py
│	│   │   └── lda_model.py
│	│   │ 
│	│   └── classification_models/
│	│       ├── __init__.py   
│	│       ├── rf_classifier.py
│	│       └──  xgb_classifier.py
│	│
│	├── visualizations/
│	│   ├── __init__.py
│	│   ├── lda_viz.py
│	│   └── tables.py
│	│
│  ├── utils/
│	│   ├── __init__.py
│	│   ├── email_text_processing.py
│	│   ├── statistical_transforms.py
│	│   └── widgets.py
│  │
│  └── config/
│      ├── __init__.py
│	    └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── models/
│   │   └── __init__.py
│   │
│   ├── visualizations/
│   │   └── __init__.py
│   │  
│   └── utils/
│       └── __init__.py
│
├── data/
│   ├── raw_data/
│   ├── unwanted_texts/
│   └── processed_data/
│	    ├── csv_format
│	    ├── data_frame
│	    └── json_format
│
├── notebooks/
│   ├── 001_tutorial_text_pruning_and_processing.ipynb
│   ├── 002_tutorial_add_embedings.ipynb
│   ├── 003_tutorial_lda_model.ipynb
│   ├── 004_tutorial_doc2vec_kmeans.ipynb
│   ├── 005_tutorial_word2vec_kmeans.ipynb
│   ├── 006_tutorial_random_forest_classification.ipynb
│   ├── 007_tutorial_xgboost_classifier.ibynb
│   └── 008_email_classifier
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

