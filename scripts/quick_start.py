#!/usr/bin/env python3
"""
POI Recommender System - Quick Start Guide

This script provides a guided introduction to training and evaluating
POI recommendation models with minimal configuration.

Usage:
    python scripts/quick_start.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_welcome():
    """Print welcome message and instructions"""
    print("=" * 60)
    print("POI RECOMMENDER SYSTEM - QUICK START")
    print("=" * 60)
    print()
    print("Welcome! This guide will help you train and evaluate")
    print("POI recommendation models using real FourSquare data.")
    print()
    print("Available Models:")
    print("1. Improved Cluster Model (Recommended)")
    print("   - Best accuracy: 21.56%")
    print("   - Real user features from FourSquare")
    print("   - High representativeness: 94.7%")
    print()
    print("2. Improved Hybrid Model")
    print("   - Best representativeness: 95.3%")
    print("   - Real user features from FourSquare")
    print("   - Good accuracy: 19.21%")
    print()
    print("3. Baseline Models (for comparison)")
    print("   - Baseline Cluster: 18.7% accuracy")
    print("   - Baseline Hybrid: 16.8% accuracy")
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
                from experiments.improved_cluster_experiment import main as train_improved_cluster
                train_improved_cluster()
                break
                
            elif choice == "2":
                print("\nTraining Improved Hybrid Model...")
                from experiments.improved_hybrid_experiment import main as train_improved_hybrid
                train_improved_hybrid()
                break
                
            elif choice == "3":
                print("\nTraining All Models...")
                from scripts.run_all_experiments import main as run_all
                run_all()
                break
                
            elif choice == "4":
                print("\nRunning Complete Pipeline...")
                from scripts.run_complete_pipeline import main as run_pipeline
                run_pipeline()
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
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print()
    print("Results are available in:")
    print("- models/          : Trained model weights")
    print("- results/metrics/ : Performance data (JSON/CSV)")
    print("- results/visualizations/ : Publication-ready plots")
    print()
    print("Key Results:")
    print("- Improved Cluster: 21.56% accuracy, 94.7% representativeness")
    print("- Improved Hybrid:  19.21% accuracy, 95.3% representativeness")
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
                return
            elif choice == "3":
                print("Goodbye!")
                return
        
        # Interactive training
        interactive_training()
        
        # Show results
        show_results()
        
    except KeyboardInterrupt:
        print("\n\nQuick start interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during quick start: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
