#!/usr/bin/env python3
"""
Comprehensive MNIST Model Training and Evaluation Script

This script trains and evaluates all model iterations (iter1.py through iter17.py)
and generates a comprehensive results table in RESULTS.md.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import sys
import os
import importlib
import traceback
from collections import OrderedDict

def setup_device():
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device

def load_data():
    """Load MNIST dataset and create data loaders."""
    # Define transform to convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=True, 
                                             transform=transform, 
                                             download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transform, 
                                            download=True)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    return train_loader, test_loader

def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_description(iteration_num, model):
    """Generate a brief description of the model architecture."""
    param_count = count_parameters(model)
    
    # Try to analyze the model structure
    model_str = str(model)
    
    if 'conv' in model_str.lower():
        conv_layers = model_str.count('Conv2d')
        fc_layers = model_str.count('Linear')
        has_dropout = 'Dropout' in model_str
        has_batchnorm = 'BatchNorm' in model_str
        
        desc = f"CNN: {conv_layers} conv + {fc_layers} FC layers"
        if has_dropout:
            desc += " + Dropout"
        if has_batchnorm:
            desc += " + BatchNorm"
    else:
        fc_layers = model_str.count('Linear')
        has_dropout = 'Dropout' in model_str
        has_batchnorm = 'BatchNorm' in model_str
        
        desc = f"MLP: {fc_layers} FC layers"
        if has_dropout:
            desc += " + Dropout"
        if has_batchnorm:
            desc += " + BatchNorm"
    
    return desc

def train_model(model, train_loader, device, num_epochs=1, learning_rate=0.001):
    """Train a model and return training metrics."""
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
    
    training_time = time.time() - start_time
    train_accuracy = 100 * epoch_correct / epoch_total
    avg_loss = running_loss / len(train_loader)
    
    return {
        'train_accuracy': train_accuracy,
        'train_loss': avg_loss,
        'training_time': training_time
    }

def test_model(model, test_loader, device):
    """Test a model and return test metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_time = time.time() - start_time
    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': avg_test_loss,
        'test_time': test_time
    }

def load_iteration_model(iteration_num):
    """Load and instantiate a model from an iteration file."""
    try:
        # Add iterations directory to path if not already there
        if 'iterations' not in sys.path:
            sys.path.insert(0, 'iterations')
            
        module_name = f'iter{iteration_num}'
        module = importlib.import_module(module_name)
        
        # Instantiate the model
        model = module.MNISTNet()
        return model, None
        
    except Exception as e:
        return None, str(e)

def main():
    """Main function to train and evaluate all models."""
    print("=" * 80)
    print("MNIST Model Training and Evaluation")
    print("=" * 80)
    
    # Setup
    device = setup_device()
    train_loader, test_loader = load_data()
    
    # Results storage
    results = []
    
    # Process each iteration
    for iteration_num in range(1, 18):  # iter1.py through iter17.py
        print(f"\n{'='*60}")
        print(f"Processing Iteration {iteration_num}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, error = load_iteration_model(iteration_num)
            if model is None:
                print(f"❌ Failed to load iter{iteration_num}: {error}")
                results.append({
                    'iteration': iteration_num,
                    'description': 'Failed to load',
                    'parameters': 0,
                    'train_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'train_loss': 0.0,
                    'test_loss': 0.0,
                    'training_time': 0.0,
                    'test_time': 0.0,
                    'error': error
                })
                continue
            
            # Get model info
            param_count = count_parameters(model)
            description = get_model_description(iteration_num, model)
            
            print(f"Model: {description}")
            print(f"Parameters: {param_count:,}")
            
            # Train model
            print("Training...")
            train_metrics = train_model(model, train_loader, device)
            print(f"✅ Training completed - Accuracy: {train_metrics['train_accuracy']:.2f}%")
            
            # Test model
            print("Testing...")
            test_metrics = test_model(model, test_loader, device)
            print(f"✅ Testing completed - Accuracy: {test_metrics['test_accuracy']:.2f}%")
            
            # Store results
            result = {
                'iteration': iteration_num,
                'description': description,
                'parameters': param_count,
                'train_accuracy': train_metrics['train_accuracy'],
                'test_accuracy': test_metrics['test_accuracy'],
                'train_loss': train_metrics['train_loss'],
                'test_loss': test_metrics['test_loss'],
                'training_time': train_metrics['training_time'],
                'test_time': test_metrics['test_time'],
                'error': None
            }
            results.append(result)
            
            print(f"📊 Summary: Train={train_metrics['train_accuracy']:.2f}%, "
                  f"Test={test_metrics['test_accuracy']:.2f}%, "
                  f"Params={param_count:,}")
            
        except Exception as e:
            print(f"❌ Error processing iter{iteration_num}: {str(e)}")
            traceback.print_exc()
            
            results.append({
                'iteration': iteration_num,
                'description': 'Error during execution',
                'parameters': 0,
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'train_loss': 0.0,
                'test_loss': 0.0,
                'training_time': 0.0,
                'test_time': 0.0,
                'error': str(e)
            })
    
    # Generate results markdown
    generate_results_markdown(results)
    
    print(f"\n{'='*80}")
    print("✅ All iterations processed! Results saved to RESULTS.md")
    print(f"{'='*80}")

def generate_results_markdown(results):
    """Generate a comprehensive markdown results file."""
    
    with open('RESULTS.md', 'w') as f:
        f.write("# MNIST Model Training Results\n\n")
        f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This document contains the results of training and testing 17 different neural network architectures on the MNIST digit classification dataset.\n\n")
        
        # Summary statistics
        successful_runs = [r for r in results if r['error'] is None]
        if successful_runs:
            best_test_acc = max(successful_runs, key=lambda x: x['test_accuracy'])
            best_train_acc = max(successful_runs, key=lambda x: x['train_accuracy'])
            most_params = max(successful_runs, key=lambda x: x['parameters'])
            least_params = min(successful_runs, key=lambda x: x['parameters'])
            
            f.write("## Summary\n\n")
            f.write(f"- **Total models trained:** {len(successful_runs)}/{len(results)}\n")
            f.write(f"- **Best test accuracy:** {best_test_acc['test_accuracy']:.2f}% (Iteration {best_test_acc['iteration']})\n")
            f.write(f"- **Best train accuracy:** {best_train_acc['train_accuracy']:.2f}% (Iteration {best_train_acc['iteration']})\n")
            f.write(f"- **Largest model:** {most_params['parameters']:,} parameters (Iteration {most_params['iteration']})\n")
            f.write(f"- **Smallest model:** {least_params['parameters']:,} parameters (Iteration {least_params['iteration']})\n\n")
        
        # Main results table
        f.write("## Detailed Results\n\n")
        f.write("| Iteration | Description | Parameters | Train Acc (%) | Test Acc (%) | Train Loss | Test Loss | Train Time (s) | Test Time (s) | Status |\n")
        f.write("|-----------|-------------|------------|---------------|--------------|------------|-----------|----------------|---------------|--------|\n")
        
        for result in results:
            status = "✅ Success" if result['error'] is None else "❌ Failed"
            f.write(f"| {result['iteration']} | {result['description']} | {result['parameters']:,} | "
                   f"{result['train_accuracy']:.2f} | {result['test_accuracy']:.2f} | "
                   f"{result['train_loss']:.4f} | {result['test_loss']:.4f} | "
                   f"{result['training_time']:.2f} | {result['test_time']:.2f} | {status} |\n")
        
        # Error details if any
        failed_runs = [r for r in results if r['error'] is not None]
        if failed_runs:
            f.write("\n## Error Details\n\n")
            for result in failed_runs:
                f.write(f"### Iteration {result['iteration']}\n")
                f.write(f"**Error:** {result['error']}\n\n")
        
        # Performance analysis
        if successful_runs:
            f.write("## Performance Analysis\n\n")
            
            # Sort by test accuracy
            sorted_by_test_acc = sorted(successful_runs, key=lambda x: x['test_accuracy'], reverse=True)
            f.write("### Top 5 Models by Test Accuracy\n\n")
            f.write("| Rank | Iteration | Test Accuracy (%) | Description | Parameters |\n")
            f.write("|------|-----------|------------------|-------------|------------|\n")
            for i, result in enumerate(sorted_by_test_acc[:5], 1):
                f.write(f"| {i} | {result['iteration']} | {result['test_accuracy']:.2f} | {result['description']} | {result['parameters']:,} |\n")
            
            # Parameter efficiency
            f.write("\n### Parameter Efficiency (Test Accuracy / Parameters)\n\n")
            efficiency_results = [(r, r['test_accuracy'] / max(r['parameters'], 1) * 1000) for r in successful_runs]
            efficiency_results.sort(key=lambda x: x[1], reverse=True)
            
            f.write("| Rank | Iteration | Efficiency Score | Test Accuracy (%) | Parameters | Description |\n")
            f.write("|------|-----------|------------------|------------------|------------|-------------|\n")
            for i, (result, efficiency) in enumerate(efficiency_results[:5], 1):
                f.write(f"| {i} | {result['iteration']} | {efficiency:.4f} | {result['test_accuracy']:.2f} | {result['parameters']:,} | {result['description']} |\n")
        
        f.write("\n## Training Configuration\n\n")
        f.write("- **Dataset:** MNIST (60,000 training, 10,000 test images)\n")
        f.write("- **Batch Size:** 64\n")
        f.write("- **Epochs:** 1\n")
        f.write("- **Learning Rate:** 0.001\n")
        f.write("- **Optimizer:** Adam\n")
        f.write("- **Loss Function:** CrossEntropyLoss\n")
        f.write("- **Data Normalization:** Mean=0.1307, Std=0.3081\n\n")
        
        f.write("## Notes\n\n")
        f.write("- All models were trained for 1 epoch for consistency and speed\n")
        f.write("- Training and test times include data loading and processing\n")
        f.write("- Parameter counts include only trainable parameters\n")
        f.write("- Efficiency score is calculated as: (Test Accuracy / Parameters) × 1000\n")

if __name__ == "__main__":
    main()