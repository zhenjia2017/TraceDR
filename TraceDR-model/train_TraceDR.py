import os
import sys
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import json

from data_process import TraceDRDataset, prepare_intermediate_data, collate_fn
from model import HeterogeneousGNN
#import evaluation
from metrics import aggregate_metrics

# set random seed
torch.manual_seed(1203)
np.random.seed(1203)

# default parameters
model_name = 'TraceDR0718'

# training parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--retrieval', action='store_true', default=False, help="enable data retrieval and preparation")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default='', help='resume path')

# data parameters
parser.add_argument('--data_dir', type=str, default='data', help="data directory")
parser.add_argument('--benchmark', type=str, default='DrugRec0716', help="benchmark name")
parser.add_argument('--intermediate_dir', type=str, default='intermediate_data', help="intermediate data directory")

# model parameters
parser.add_argument('--emb_dimension', type=int, default=768, help="embedding dimension")
parser.add_argument('--num_layers', type=int, default=3, help="number of GNN layers")
parser.add_argument('--dropout', type=float, default=0.0, help="dropout rate")
parser.add_argument('--max_entities', type=int, default=100, help="max entities")
parser.add_argument('--max_evidences', type=int, default=50, help="max evidences")
parser.add_argument('--encoder_lm', type=str, default='BERT', help="encoder language model")
parser.add_argument('--encoder_linear', action='store_true', default=False, help="use linear encoder")

# training parameters
parser.add_argument('--epochs', type=int, default=5, help="number of epochs")
parser.add_argument('--train_batch_size', type=int, default=1, help="training batch size")
parser.add_argument('--eval_batch_size', type=int, default=1, help="evaluation batch size")
parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
parser.add_argument('--clipping_max_norm', type=float, default=1.0, help="gradient clipping max norm")

# data processing parameters
parser.add_argument('--tsf_delimiter', type=str, default='||', help="TSF delimiter")
parser.add_argument('--max_pos_evidences', type=int, default=10, help="max positive evidences")

# test output file parameters
parser.add_argument('--test_output', type=str, default='output/test_results', help="test output directory")

args = parser.parse_args()

def move_to_cuda(batch):
    """Move batch to GPU"""
    if torch.cuda.is_available():
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()
    return batch

def eval_model(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    qa_metrics_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = move_to_cuda(batch)
            
            output = model(batch, train=False)
            loss = output["loss"]
            total_loss += loss.item()
            
            # collect QA metrics
            qa_metrics = output["qa_metrics"]
            qa_metrics_list.extend(qa_metrics)
    
    avg_loss = total_loss / len(data_loader)
    
    # aggregate QA metrics
    avg_qa_metrics = aggregate_metrics(qa_metrics_list)
    
    return avg_loss, avg_qa_metrics

def save_evaluation_results(save_dir, epoch, metrics, loss, is_best=False):
    """Save evaluation results"""
    results = {
        "epoch": epoch,
        "loss": loss,
        "metrics": metrics,
        "timestamp": time.time()
    }
    
    if is_best:
        # save best results
        best_results_path = os.path.join(save_dir, "best_evaluation_results.json")
        with open(best_results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # save all evaluation results to history record
    history_path = os.path.join(save_dir, "evaluation_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(results)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

def test_model_with_detailed_output():
    """Test model and save detailed results"""
    print("start testing with detailed output")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # create output directory
    output_dir = args.test_output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load test data
    test_path = os.path.join(args.intermediate_dir, args.benchmark, "test.pkl")
    if not os.path.exists(test_path):
        test_path = prepare_intermediate_data(args.data_dir, args.benchmark, args.intermediate_dir, 'test')
    
    test_dataset = TraceDRDataset(
        data_path=test_path,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        train=False,
        tsf_delimiter=args.tsf_delimiter
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Test dataset size: {len(test_dataset)}")

    model = HeterogeneousGNN(
        emb_dimension=args.emb_dimension,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        encoder_lm=args.encoder_lm,
        encoder_linear=args.encoder_linear
    )
    
    # load model
    if args.resume_path:
        model_path = args.resume_path
    else:
        model_path = os.path.join("saved", args.model_name, "best_model.pt")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # run inference and collect detailed results
    model.eval()
    total_loss = 0.0
    qa_metrics_list = []
    detailed_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch = move_to_cuda(batch)
            
            output = model(batch, train=False)
            loss = output["loss"]
            total_loss += loss.item()
            
            # collect QA metrics
            qa_metrics = output["qa_metrics"]
            qa_metrics_list.extend(qa_metrics)
            
            # collect detailed results (refer to gnn-multitask implementation)
            for i in range(len(batch["tsf"])):
                # extract ranked_answers and top_evidences
                ranked_answers = []
                top_evidences = []
                
                if "answer_predictions" in output and i < len(output["answer_predictions"]):
                    ranked_answers = output["answer_predictions"][i]
                
                if "evidence_predictions" in output and i < len(output["evidence_predictions"]):
                    top_evidences = output["evidence_predictions"][i]
                
                instance_result = {
                    "instance_idx": i,
                    "question": batch["question"][i],
                    "ranked_answers": ranked_answers,
                    "top_evidences": top_evidences,
                    "qa_metrics": qa_metrics[i] if i < len(qa_metrics) else {}
                }
                detailed_results.append(instance_result)
    
    avg_loss = total_loss / len(test_loader)
    avg_qa_metrics = aggregate_metrics(qa_metrics_list)
    
    # save test results
    test_metrics = {
        "test_loss": avg_loss,
        "test_metrics": avg_qa_metrics,
        "model_path": model_path,
        "test_config": {
            "batch_size": args.eval_batch_size,
            "model_name": args.model_name,
            "benchmark": args.benchmark
        },
        "timestamp": time.time()
    }
    
    # save evaluation metrics result file
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # save detailed evaluation record file
    detailed_path = os.path.join(output_dir, "test_detailed_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print("Test Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    if avg_qa_metrics:
        print(f"  Test Metrics: {avg_qa_metrics}")
    
    print(f"Results saved to:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Detailed Results: {detailed_path}")

def train_model():
    """Train model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a save directory
    save_dir = os.path.join("saved", args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Step 1: Preparation of intermediate data
    if args.retrieval:
        print("Step 1: Preparing intermediate data...")
        step1_start = time.time()
        
        train_path = prepare_intermediate_data(args.data_dir, args.benchmark, args.intermediate_dir, 'train')
        dev_path = prepare_intermediate_data(args.data_dir, args.benchmark, args.intermediate_dir, 'dev')
        test_path = prepare_intermediate_data(args.data_dir, args.benchmark, args.intermediate_dir, 'test')
        
        print(f"Time taken (Data Preparation): {time.time() - step1_start:.2f} seconds")
    else:
        print("Step 1: Skipping data preparation, using existing intermediate data...")
        # use existing intermediate data paths
        train_path = os.path.join(args.intermediate_dir, args.benchmark, "train.pkl")
        dev_path = os.path.join(args.intermediate_dir, args.benchmark, "dev.pkl")
        test_path = os.path.join(args.intermediate_dir, args.benchmark, "test.pkl")
    
    # Step 2: Load data
    print("Step 2: Loading datasets...")
    train_dataset = TraceDRDataset(
        data_path=train_path,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        train=True,
        tsf_delimiter=args.tsf_delimiter,
        max_pos_evidences=args.max_pos_evidences
    )
    
    dev_dataset = TraceDRDataset(
        data_path=dev_path,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        train=False,
        tsf_delimiter=args.tsf_delimiter,
        max_pos_evidences=args.max_pos_evidences
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    
    # Step 3: model initialization
    print("Step 3: Initializing model...")
    model = HeterogeneousGNN(
        emb_dimension=args.emb_dimension,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        encoder_lm=args.encoder_lm,
        encoder_linear=args.encoder_linear
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")
    
    # Step 4: Set up the optimiser
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Step 5: train
    print("Step 4: Training...")
    step2_start = time.time()
    
    best_metric = -1
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = move_to_cuda(batch)
            
            optimizer.zero_grad()
            output = model(batch, train=True)
            loss = output["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_max_norm)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # eval
        dev_loss, dev_metrics = eval_model(model, dev_loader, device)
        
        # print results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")
        if dev_metrics:
            print(f"  Dev Metrics: {dev_metrics}")
        
        # save the best model
        decisive_metric = "p_at_1"  # use p_at_1 as decisive metric
        current_metric = dev_metrics.get(decisive_metric, 0.0)
        
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            
            # save model
            model_path = os.path.join(save_dir, f"best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model with {decisive_metric}: {current_metric:.4f}")
        
        # save evaluation results (save every epoch)
        save_evaluation_results(save_dir, epoch, dev_metrics, dev_loss, is_best)
    
    print(f"Time taken (Training): {time.time() - step2_start:.2f} seconds")
    print(f"Best epoch: {best_epoch+1}, Best {decisive_metric}: {best_metric:.4f}")

def test_model():
    """test model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load test data
    test_path = os.path.join(args.intermediate_dir, args.benchmark, "test.pkl")
    if not os.path.exists(test_path):
        test_path = prepare_intermediate_data(args.data_dir, args.benchmark, args.intermediate_dir, 'test')
    
    test_dataset = TraceDRDataset(
        data_path=test_path,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        train=False,
        tsf_delimiter=args.tsf_delimiter
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Test dataset size: {len(test_dataset)}")

    model = HeterogeneousGNN(
        emb_dimension=args.emb_dimension,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        encoder_lm=args.encoder_lm,
        encoder_linear=args.encoder_linear
    )
    
    # load model
    if args.resume_path:
        model_path = args.resume_path
    else:
        model_path = os.path.join("saved", args.model_name, "best_model.pt")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # test
    test_loss, test_metrics = eval_model(model, test_loader, device)
    
    print("Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    if test_metrics:
        print(f"  Test Metrics: {test_metrics}")

def main():
    if args.eval:
        # test_model()
        test_model_with_detailed_output()
    else:
        train_model()

if __name__ == '__main__':
    main() 