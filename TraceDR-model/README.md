# TraceDR
This is the code for the paper *Traceable Drug Recommendation over Medical Knowledge Graphs*.

TraceDR is a drug recommendation system that leverages heterogeneous graph neural networks (GNN) to provide personalized and safe medication recommendations based on patient medical history and knowledge graph information.


## Architecture

The system consists of four main components:

1. **TraceDRDataset**: Data processing and graph construction.
2. **FullEncoder**: BERT-based text encoder that processes patient information, entities, and evidences
3. **HeterogeneousGNN**: Multi-layer graph neural network that performs message passing between entities and evidences
4. **MultitaskBilinearAnswering**: Bilinear answering module for drug recommendation and evidence ranking

## Project Structure

```
TraceDR/
├── train_TraceDR.py          # Main training and evaluation script
├── model.py                  # Model architecture (HeterogeneousGNN, FullEncoder, etc.)
├── data_process.py           # Data preprocessing and dataset definition
├── metrics.py                # Evaluation metrics (P@1, MRR, Hit@5, Jaccard, F1, etc.)
├── bm25_es.py               # BM25 retrieval implementation
├── requirements.txt          # Project dependencies
├── data/                     # Data directory
├── intermediate_data/        # Processed intermediate data
├── output/                   # Test results and outputs
├── saved/                    # Saved model checkpoints
└── baseline/                 # Baseline model
```

## Dataset
Download the [DrugRec data](https://drive.google.com/file/d/1JFopwcckmrtWtK01XQTbK5huodeaaaOR/view?usp=drive_link) and put it in ./data

This dataset was created using the [DrugRecSynthesis](https://anonymous.4open.science/r/DrugRecSynthesis-1E7D)


## Training

```bash
python train_TraceDR.py \
    --data_dir data \
    --benchmark DrugRec0716 \
    --intermediate_dir intermediate_data \
    --emb_dimension 768 \
    --num_layers 3 \
    --epochs 5 \
    --train_batch_size 1 \
    --lr 1e-5 \
    --model_name TraceDR_v1
```

### Evaluation

```bash
python train_TraceDR.py \
    --eval \
    --model_name TraceDR_v1 \
    --test_output output/test_results
```

### Testing with Detailed Output

```bash
python train_TraceDR.py \
    --eval \
    --model_name TraceDR_v1 \
    --resume_path saved/TraceDR_v1/best_model.pt \
    --test_output output/detailed_results
```

## Configuration

### Data Parameters
- `--data_dir`: Original data directory (default: `data`)
- `--benchmark`: Dataset name (default: `DrugRec0716`)
- `--intermediate_dir`: Intermediate data directory (default: `intermediate_data`)
- `--retrieval`: Enable data retrieval and preparation (flag)

### Model Parameters
- `--emb_dimension`: Embedding dimension (default: `768`)
- `--num_layers`: Number of GNN layers (default: `3`)
- `--dropout`: Dropout rate (default: `0.0`)
- `--max_entities`: Maximum number of entities (default: `100`)
- `--max_evidences`: Maximum number of evidences (default: `50`)
- `--encoder_lm`: Encoder language model (default: `BERT`)
- `--encoder_linear`: Use linear encoder (flag)

### Training Parameters
- `--epochs`: Number of training epochs (default: `5`)
- `--train_batch_size`: Training batch size (default: `1`)
- `--eval_batch_size`: Evaluation batch size (default: `1`)
- `--lr`: Learning rate (default: `1e-5`)
- `--weight_decay`: Weight decay (default: `0.01`)
- `--clipping_max_norm`: Gradient clipping max norm (default: `1.0`)

### Data Processing Parameters
- `--tsf_delimiter`: query delimiter (default: `||`)
- `--max_pos_evidences`: Maximum positive evidences (default: `10`)

## Evaluation Metrics

The system evaluates performance using multiple metrics:

- **P@k**: Precision at top-K
- **R@K**: Recall at top-K
- **f1@k**: F1 at top-K
- **Jaccard**: Jaccard similarity coefficient
- **F1@K**: F1 score at top K
- **Answer Presence**: Presence of correct answers in candidates
- **DDI Rate**: Drug-Drug Interaction safety metrics

## Training Pipeline

1. **Data Preparation**: 
   - BM25-based retrieval from medical knowledge graph
   - Patient information processing and graph construction
   - Intermediate data generation

2. **Model Training**:
   - Heterogeneous graph construction with entities and evidences
   - Multi-layer message passing between graph nodes
   - Joint training of answer prediction and evidence ranking

3. **Evaluation**:
   - Performance measurement using multiple metrics
   - Best model selection based on P@1 score
   - Detailed result analysis and output file


### Output Format
```json
{
  "test_metrics": {
    "test_loss": 0.1234,
    "test_metrics": {
      "p_at_1": 0.85,
      "mrr": 0.78,
      "h_at_5": 0.92
    }
  },
  "detailed_results": [
    {
      "question": "patient_query",
      "ranked_answers": [...],
      "top_evidences": [...],
      "qa_metrics": {...}
    }
  ]
}
```

## Dependencies

```
torch>=1.8.0
transformers>=4.0.0
py2neo>=2021.0
jieba>=0.42.0
scikit-learn>=0.24.0
python-Levenshtein>=0.12.0
tqdm>=4.60.0
numpy>=1.20.0
pandas>=1.3.0
neo4j>=4.4.0
```
