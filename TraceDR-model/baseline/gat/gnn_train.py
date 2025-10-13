import argparse
import json
import numpy as np
import os
import pickle
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from collections import defaultdict

from model import GAT
from data_process import DatasetGNN, collate_fn
from metrics import *

# Set random seeds
torch.manual_seed(1203)
np.random.seed(1203)
random.seed(1203)

# Default model configuration
model_name = 'GAT_DrugRec'
START_DATE = time.strftime("%y-%m-%d_%H-%M", time.localtime())
random_integer = random.randint(1, 1000)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GAT Training')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default=model_name, help='Model name for saving')
    parser.add_argument('--emb_dimension', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_entities', type=int, default=50, help='Maximum number of entities')
    parser.add_argument('--max_evidences', type=int, default=100, help='Maximum number of evidences')
    
    # Encoder parameters
    parser.add_argument('--max_input_length_sr', type=int, default=128, help='Max input length for structured representation')
    parser.add_argument('--max_input_length_ev', type=int, default=64, help='Max input length for evidences')
    parser.add_argument('--max_input_length_ent', type=int, default=32, help='Max input length for entities')
    parser.add_argument('--encoder_linear', action='store_true', default=False, help='Use linear encoder layer')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clipping_max_norm', type=float, default=1.0, help='Gradient clipping max norm')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='../../intermediate_data/DrugRec0716/', help='Data directory')
    parser.add_argument('--train_file', type=str, default='train.pkl', help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.pkl', help='Development data file')
    parser.add_argument('--test_file', type=str, default='test.pkl', help='Test data file')
    
    # Model saving
    parser.add_argument('--model_dir', type=str, default='saved', help='Model save directory')
    parser.add_argument('--decisive_metric', type=str, default='p_at_1', help='Decisive metric for model saving')
    
    # Mode
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode')
    parser.add_argument('--resume_path', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def move_to_cuda(obj):
    """Move tensors to CUDA if available."""
    if torch.cuda.is_available():
        for key, value in obj.items():
            if isinstance(obj[key], torch.Tensor):
                obj[key] = obj[key].cuda()


def evaluate(model, data_loader, epoch, args):
    """Evaluate the model."""
    print('\nEvaluating...')
    model.eval()
    
    total_loss = 0.0
    all_qa_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating')):
            move_to_cuda(batch)
            
            output = model(batch, train=False)
            
            if output["loss"] is not None:
                total_loss += output["loss"].item()
            
            if output["qa_metrics"]:
                all_qa_metrics.extend(output["qa_metrics"])
            
            # Free GPU memory
            del batch

    # Calculate average metrics
    avg_loss = total_loss / len(data_loader) if data_loader else 0.0
    
    if all_qa_metrics:
        avg_metrics = {}
        metric_keys = ["p_at_1", "mrr", "h_at_5", "answer_presence", "jaccard", 
                      "avg_p", "avg_re", "avg_f1", "DDI_rate", "DDI_rate@1", 
                      "CMS", "group_rate"]
        
        for key in metric_keys:
            if key in all_qa_metrics[0]:
                avg_metrics[key] = np.mean([m[key] for m in all_qa_metrics])
        
        avg_metrics["num_questions"] = len(all_qa_metrics)
    else:
        avg_metrics = {}

    print(f'Epoch {epoch}: Loss: {avg_loss:.4f}')
    print(f'Metrics: {avg_metrics}')
    
    return avg_loss, avg_metrics


def train_epoch(model, data_loader, optimizer, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
        move_to_cuda(batch)
        
        # Forward pass
        output = model(batch, train=True)
        loss = output["loss"]
        
        if loss is not None:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_max_norm)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Free GPU memory
        del batch
    
    avg_loss = total_loss / len(data_loader) if data_loader else 0.0
    print(f'Epoch {epoch}: Training Loss: {avg_loss:.4f}')
    
    return avg_loss


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create model directory
    model_save_dir = os.path.join(args.model_dir, args.model_name)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    test_path = os.path.join(args.data_dir, args.test_file)
    
    if not args.eval:
        train_data = DatasetGNN(
            data_path=train_path,
            max_entities=args.max_entities,
            max_evidences=args.max_evidences,
            train=True
        )
        train_loader = DataLoader(
            train_data, 
            batch_size=args.train_batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        print(f'Training data size: {len(train_data)}')
    
    dev_data = DatasetGNN(
        data_path=dev_path,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        train=False
    )
    dev_loader = DataLoader(
        dev_data, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    print(f'Development data size: {len(dev_data)}')
    
    # Initialize model
    model = GAT(
        emb_dimension=args.emb_dimension,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
        max_input_length_sr=args.max_input_length_sr,
        max_input_length_ev=args.max_input_length_ev,
        max_input_length_ent=args.max_input_length_ent,
        encoder_linear=args.encoder_linear
    )
    
    # Move model to device
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')
    
    # Load checkpoint if provided
    if args.resume_path and os.path.exists(args.resume_path):
        print(f'Loading checkpoint from {args.resume_path}')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.resume_path))
        else:
            model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    
    # Evaluation mode
    if args.eval:
        test_data = DatasetGNN(
            data_path=test_path,
            max_entities=args.max_entities,
            max_evidences=args.max_evidences,
            train=False
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        print(f'Test data size: {len(test_data)}')
        
        _, test_metrics = evaluate(model, test_loader, 0, args)
        print('Test Results:', test_metrics)
        return
    
    # Training mode
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_metric = -1
    best_epoch = 0
    history = defaultdict(list)
    
    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, args)
        
        # Evaluate
        dev_loss, dev_metrics = evaluate(model, dev_loader, epoch, args)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        
        for key, value in dev_metrics.items():
            if isinstance(value, (int, float)):
                history[f'dev_{key}'].append(value)
        
        # Save best model
        decisive_metric_value = dev_metrics.get(args.decisive_metric, 0.0)
        if decisive_metric_value > best_metric:
            best_metric = decisive_metric_value
            best_epoch = epoch
            
            # Save model
            model_path = os.path.join(model_save_dir, 'best_model.bin')
            torch.save(model.state_dict(), model_path)
            
            # Save timestamped model
            timestamped_path = os.path.join(
                model_save_dir, 
                f'model-{START_DATE}-{random_integer}-epoch{epoch}.bin'
            )
            torch.save(model.state_dict(), timestamped_path)
            
            # Save config
            config_path = os.path.join(model_save_dir, f'config-{START_DATE}-{random_integer}.json')
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            
            print(f'New best model saved! {args.decisive_metric}: {decisive_metric_value:.4f}')
        
        # Time tracking
        epoch_time = (time.time() - start_time) / 60
        remaining_time = epoch_time * (args.epochs - epoch) / 60
        print(f'Epoch time: {epoch_time:.2f}m, Estimated remaining: {remaining_time:.2f}h\n')
    
    # Save final model and history
    final_model_path = os.path.join(model_save_dir, 'final_model.bin')
    torch.save(model.state_dict(), final_model_path)
    
    history_path = os.path.join(model_save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)
    
    print(f'Training completed! Best epoch: {best_epoch}, Best {args.decisive_metric}: {best_metric:.4f}')


if __name__ == '__main__':
    main() 