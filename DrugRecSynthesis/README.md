# DrugRecSynthesis

A  system for generating synthetic patient data with drug recommendations, leveraging knowledge graphs and Large Language Models (LLMs) to generate Reasonable drug recommendation cases for research and development purposes.

## Overview

This system generates synthetic patient datasets containing:
- Patient demographics (age, gender, special populations)
- Medical diagnoses and symptoms  
- Drug recommendations
- Medical history and concomitant medication
- Drug interactions and contraindications
- Allergy information

The generated data follows medical logic and constraints to ensure realistic scenarios while maintaining patient privacy through synthetic generation.

## System Architecture

### Core Components

1. **Synthetic Data Generator (`Synthetic` class)**: Main orchestrator for data generation
2. **LLM API Interface (`LLMAPI`)**: Handles symptom generation and data validation
3. **Drug Review System (`DrugReviewSystem`)**: Validates drug recommendations using medical knowledge
4. **Neo4j Knowledge Graph**: Stores medical knowledge including drug information, interactions, and contraindications
5. **Data Analyzer (`DataAnalyzer`)**: Provides statistical analysis of generated data

### Knowledge Sources
The knowledge graph used in this system comes from: https://github.com/zhenjia2017/RDUKG


## Installation

### Prerequisites

```bash
# Python 3.7+
pip install py2neo pandas tqdm
```

### Dependencies

- **Neo4j Database**: Medical knowledge graph
- **LLM API Access**: For symptom generation and validation
- **Required Data Files**:
  - `diagnosis_filename`: Medical conditions and treatments mapping
  - `geography_file_path`: Age probability distributions
  - `allergen_filename`: Drug allergen database
  - `drug_interaction_analysis_dict.pkl`: Drug interaction data
  - `drugMsg_linux_dict.pkl`: Drug information database

### Configuration

Configure the system parameters in `args.py`:

```python
# Generation parameters
people_num = 21000          # Number of patients to generate
upper_limit = 2           # Maximum patients per diagnosis
consider_coverage = 1      # Enable diagnosis coverage monitoring

# Probability settings
allergen_prob = 0.2        # Probability of drug allergies
medhistory_prob = 0.3      # Probability of medical history
liver_prob = 0.03          # Probability of liver dysfunction
kidney_prob = 0.10         # Probability of kidney dysfunction
pregnant_prob = 0.035       # Probability of pregnancy (fertile females)
lactation_prob = 0.057      # Probability of lactation (fertile females)

# File paths
out_doc = "DrugRec"  # Output directory
history_doc = "previous"    # Historical data directory (for continuation)
```

## Usage

### Basic Generation

```python
from synthetic_refactored import Synthetic

# Initialize generator
generator = Synthetic(
    neo4j_uri="http://localhost:7474",
    api_key="your-llm-api-key",
    model="qwen-max"
)

# Generate 1000 synthetic patients
patients = generator.generate_people_data(21000)
```

### Continuation Mode

```python
# Continue from existing data
# Set arg.history_data = 1 and arg.history_doc = "previous_output"
patients = generator.generate_people_data(500)  # Adds 500 more patients
```

### Loading Generated Data

```python
# Load from pickle format
patients = generator.load_people_data("pkl")

# Load from JSON format  
patients = generator.load_people_data("json")

# Get data summary
summary = Synthetic.get_data_summary(patients)
print(summary)
```

## Data Structure

### Patient Record Schema

```json
{
  "id": "string",
  "age": "integer", 
  "gender": "string (男/女)",
  "group": ["string array - population groups"],
  "allergen": ["string array - drug allergies"],
  "diagnosis": ["string array - diagnosis"],
  "symptom": ["string array - patient symptoms"],
  "medicine": [
    {
      "drugid": "string",
      "name": "string", 
      "CMAN": "string",
      "treat": [{"treat_id": "string", "treat": "string"}],
      "caution": [{"crowd": "string", "caution_level": "string"}],
      "ingredients": [{"ingredient_id": "string", "ingredient": "string"}],
      "interaction": [{"interaction_id": "string", "interaction": "string"}]
    }
  ],
  "antecedents": ["string array - medical history"],
  "on_medicine": ["array - concurrent medications"]
}
```

### Population Groups

- `儿童` (Children): Age < 12
- `青少年` (Adolescents): Age 12-17  
- `成人` (Adults): Age 18-64
- `老年人` (Elderly): Age ≥ 65
- `肝功能不全` (Liver dysfunction)
- `肾功能不全` (Kidney dysfunction)
- `孕妇` (Pregnant)
- `哺乳期` (Lactating)

## Output Files

The system generates several output files:

```
output/{out_doc}/
├── people_data.pkl                    # Main dataset (pickle format)
├── people_data.json                   # Main dataset (JSON format)
├── used_diagnosis_dict.json           # Diagnosis usage statistics
├── diagnosis_coverage_report.json     # Coverage analysis report
└── LLM_cache.pkl                     # LLM API response cache
```

**Note**: This system generates synthetic data for research purposes only. Do not use generated data for actual medical decisions or patient care. 