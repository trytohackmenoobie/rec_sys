#!/usr/bin/env python3
"""
DualPOI - Quick Start Guide

This script provides a guided introduction to training and evaluating
POI recommendation models with minimal configuration.

Usage:
    python scripts/quick_start.py
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dualpoi.config import Config

def print_welcome():
    """Print welcome message and instructions"""
    print("=" * 60)
    print("DUALPOI - QUICK START")
    print("=" * 60)
    print()
    print("Welcome! This guide will help you train and evaluate")
    print("POI recommendation models using real FourSquare data.")
    print()
    
    # Load metrics from experimental results
    config = Config()
    results_json_path = os.path.join(config.BASE_DIR, 'results', 'metrics', 'experimental_results.json')
    
    models_data = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)
                models_data = results_data.get('models', {})
        except Exception:
            pass
    
    improved_cluster = models_data.get('improved_cluster', {})
    improved_cluster_acc = improved_cluster.get('accuracy', 0.0) * 100 if improved_cluster else 0.0
    improved_cluster_repr = improved_cluster.get('representativeness', 0.0) * 100 if improved_cluster else 0.0
    
    improved_hybrid = models_data.get('improved_hybrid', {})
    improved_hybrid_acc = improved_hybrid.get('accuracy', 0.0) * 100 if improved_hybrid else 0.0
    improved_hybrid_repr = improved_hybrid.get('representativeness', 0.0) * 100 if improved_hybrid else 0.0
    
    baseline_cluster = models_data.get('baseline_cluster', {})
    baseline_cluster_acc = baseline_cluster.get('accuracy', 0.0) * 100 if baseline_cluster else 0.0
    
    baseline_hybrid = models_data.get('baseline_hybrid', {})
    baseline_hybrid_acc = baseline_hybrid.get('accuracy', 0.0) * 100 if baseline_hybrid else 0.0
    
    print("Available Models:")
    print("1. Improved Cluster Model (Recommended)")
    if improved_cluster_acc > 0:
        print(f"   - Best accuracy: {improved_cluster_acc:.2f}%")
    else:
        print("   - Best accuracy: (not available)")
    print("   - Real user features from FourSquare")
    print("   - Neighbor-aware semantic clustering")
    if improved_cluster_repr > 0:
        print(f"   - High representativeness: {improved_cluster_repr:.2f}%")
    else:
        print("   - High representativeness: (not available)")
    print()
    print("2. Improved Hybrid Model")
    if improved_hybrid_repr > 0:
        print(f"   - Best representativeness: {improved_hybrid_repr:.2f}%")
    else:
        print("   - Best representativeness: (not available)")
    print("   - Real user features from FourSquare")
    if improved_hybrid_acc > 0:
        print(f"   - Good accuracy: {improved_hybrid_acc:.2f}%")
    else:
        print("   - Good accuracy: (not available)")
    print()
    print("3. Baseline Models (for comparison)")
    if baseline_cluster_acc > 0:
        print(f"   - Baseline Cluster: {baseline_cluster_acc:.2f}% accuracy")
    else:
        print("   - Baseline Cluster: (metrics not available)")
    if baseline_hybrid_acc > 0:
        print(f"   - Baseline Hybrid: {baseline_hybrid_acc:.2f}% accuracy")
    else:
        print("   - Baseline Hybrid: (metrics not available)")
    print()

def interactive_training():
    """Interactive model training interface"""
    print("TRAINING OPTIONS:")
    print("-" * 30)
    print("1. Train Improved Cluster Model (recommended)")
    print("2. Train Improved Hybrid Model")
    print("3. Train All Models")
    print("4. Run Complete Pipeline")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                print("\nTraining Improved Cluster Model...")
                try:
                    from experiments.improved_cluster_experiment import main as train_improved_cluster
                    train_improved_cluster()
                except Exception as e:
                    print(f"Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                break
                
            elif choice == "2":
                print("\nTraining Improved Hybrid Model...")
                try:
                    from experiments.improved_hybrid_experiment import main as train_improved_hybrid
                    train_improved_hybrid()
                except Exception as e:
                    print(f"Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                break
                
            elif choice == "3":
                print("\nTraining All Models...")
                try:
                    from scripts.run_all_experiments import main as run_all
                    run_all()
                except Exception as e:
                    print(f"Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                break
                
            elif choice == "4":
                print("\nRunning Complete Pipeline...")
                try:
                    from scripts.run_complete_pipeline import main as run_pipeline
                    run_pipeline()
                except Exception as e:
                    print(f"Error during pipeline execution: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                break
                
            elif choice == "5":
                print("Goodbye!")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

def show_results():
    """Show training results and next steps"""
    try:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print()
        print("Results are available in:")
        print("- models/          : Trained model weights")
        print("- results/metrics/ : Performance data (JSON/CSV)")
        print("- results/visualizations/ : Publication-ready plots")
        print()
        # Load metrics from experimental results
        config = Config()
        results_json_path = os.path.join(config.BASE_DIR, 'results', 'metrics', 'experimental_results.json')
        
        models_data = {}
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    results_data = json.load(f)
                    models_data = results_data.get('models', {})
            except Exception as e:
                print(f"Warning: Could not load results from {results_json_path}: {e}")
        
        improved_cluster = models_data.get('improved_cluster', {})
        improved_cluster_acc = improved_cluster.get('accuracy', 0.0) * 100 if improved_cluster else 0.0
        improved_cluster_repr = improved_cluster.get('representativeness', 0.0) * 100 if improved_cluster else 0.0
        
        improved_hybrid = models_data.get('improved_hybrid', {})
        improved_hybrid_acc = improved_hybrid.get('accuracy', 0.0) * 100 if improved_hybrid else 0.0
        improved_hybrid_repr = improved_hybrid.get('representativeness', 0.0) * 100 if improved_hybrid else 0.0
        
        print("Key Results:")
        if improved_cluster_acc > 0 and improved_cluster_repr > 0:
            print(f"- Improved Cluster: {improved_cluster_acc:.2f}% accuracy, {improved_cluster_repr:.2f}% representativeness")
        else:
            print("- Improved Cluster: (metrics not available)")
        if improved_hybrid_acc > 0 and improved_hybrid_repr > 0:
            print(f"- Improved Hybrid:  {improved_hybrid_acc:.2f}% accuracy, {improved_hybrid_repr:.2f}% representativeness")
        else:
            print("- Improved Hybrid:  (metrics not available)")
        print()
        print("Next Steps:")
        print("1. View visualizations: results/visualizations/")
        print("2. Check results: results/metrics/")
        print("3. Read documentation: docs/")
        print()
        print("For advanced usage:")
        print("- Train specific model: python scripts/train_model.py --model improved_cluster")
        print("- Run experiments: python scripts/run_all_experiments.py")
        print("- Generate reports: python scripts/generate_all_visualizations.py")
    except Exception as e:
        print(f"Error displaying results: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main quick start interface"""
    try:
        print_welcome()
        
        # Check if results already exist
        results_dir = project_root / "results" / "metrics"
        if results_dir.exists() and list(results_dir.glob("*.json")):
            print("EXISTING RESULTS DETECTED")
            print("-" * 30)
            print("Previous training results found. You can:")
            print("1. View existing results")
            print("2. Retrain models")
            print("3. Exit")
            print()
            
            choice = input("Select option (1-3): ").strip()
            if choice == "1":
                show_results()
                print("\n" + "=" * 60)
                print("Quick start completed. Goodbye!")
                print("=" * 60)
                return
            elif choice == "3":
                print("Goodbye!")
                return
        
        # Interactive training
        interactive_training()
        
        # Show results
        show_results()
        print("\n" + "=" * 60)
        print("Quick start completed. Goodbye!")
        print("=" * 60)
        return
        
    except KeyboardInterrupt:
        print("\n\nQuick start interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during quick start: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
