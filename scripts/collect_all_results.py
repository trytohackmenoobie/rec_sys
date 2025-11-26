"""
Results Collection and Processing Script


This script processes raw experiment results and creates structured
summary files for visualization and documentation generation.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_raw_results():
    """Load raw experiment results from JSON file"""
    results_file = project_root / "results" / "metrics" / "raw_experiment_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Raw results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def process_experiment_results(raw_data):
    """Process raw results into structured format"""
    
    summary = raw_data.get("experiment_summary", {})
    processed_results = {
        "metadata": {
            "generation_time": datetime.now().isoformat(),
            "total_experiments": summary.get("total_experiments", 0),
            "successful_experiments": summary.get("successful_experiments", 0),
            "total_execution_time": summary.get("total_execution_time", 0.0)
        },
        "models": {}
    }
    
    # Process each experiment
    experiments = raw_data.get("experiments", {})
    for model_key, exp_data in experiments.items():
        if exp_data.get("status") == "success":
            results = exp_data.get("results", {})
            
            processed_results["models"][model_key] = {
                "model_name": exp_data.get("experiment_name", model_key),
                "status": "success",
                "accuracy": results.get("accuracy", 0.0),
                "representativeness": results.get("representativeness", 0.0),
                "hits_at_3": results.get("hits_at_3", 0.0),
                "hits_at_5": results.get("hits_at_5", 0.0),
                "execution_time": exp_data.get("execution_time", 0.0),
                "timestamp": exp_data.get("timestamp", datetime.now().isoformat())
            }
        else:
            processed_results["models"][model_key] = {
                "model_name": exp_data.get("experiment_name", model_key),
                "status": "failed",
                "error": exp_data.get("error", "Unknown error"),
                "execution_time": exp_data.get("execution_time", 0.0),
                "timestamp": exp_data.get("timestamp", datetime.now().isoformat())
            }
    
    return processed_results

def create_performance_summary(processed_results):
    """Create performance summary statistics"""
    
    successful_models = {
        k: v for k, v in processed_results["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        return {"error": "No successful experiments to summarize"}
    
    # Calculate statistics
    accuracies = [model["accuracy"] for model in successful_models.values()]
    representativeness_scores = [model["representativeness"] for model in successful_models.values()]
    hits_at_3_scores = [model["hits_at_3"] for model in successful_models.values()]
    hits_at_5_scores = [model["hits_at_5"] for model in successful_models.values()]
    
    summary = {
        "accuracy": {
            "mean": sum(accuracies) / len(accuracies),
            "min": min(accuracies),
            "max": max(accuracies),
            "best_model": max(successful_models.items(), key=lambda x: x[1]["accuracy"])[0]
        },
        "representativeness": {
            "mean": sum(representativeness_scores) / len(representativeness_scores),
            "min": min(representativeness_scores),
            "max": max(representativeness_scores),
            "best_model": max(successful_models.items(), key=lambda x: x[1]["representativeness"])[0]
        },
        "hits_at_3": {
            "mean": sum(hits_at_3_scores) / len(hits_at_3_scores),
            "min": min(hits_at_3_scores),
            "max": max(hits_at_3_scores)
        },
        "hits_at_5": {
            "mean": sum(hits_at_5_scores) / len(hits_at_5_scores),
            "min": min(hits_at_5_scores),
            "max": max(hits_at_5_scores)
        }
    }
    
    return summary

def create_comparison_table(processed_results):
    """Create model comparison table"""
    
    successful_models = {
        k: v for k, v in processed_results["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        return []
    
    comparison_data = []
    for model_key, model_data in successful_models.items():
        comparison_data.append({
            "Model": model_data["model_name"],
            "Accuracy": f"{model_data['accuracy']:.3f}",
            "Representativeness": f"{model_data['representativeness']:.3f}",
            "Hits@3": f"{model_data['hits_at_3']:.3f}",
            "Hits@5": f"{model_data['hits_at_5']:.3f}",
            "Execution Time (s)": f"{model_data['execution_time']:.1f}"
        })
    
    return comparison_data

def save_results(processed_results, performance_summary, comparison_table):
    """Save all processed results to files"""
    
    results_dir = project_root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed results
    processed_file = results_dir / "experimental_results.json"
    with open(processed_file, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    # Save performance summary
    summary_file = results_dir / "performance_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    # Save comparison table as CSV
    if comparison_table:
        df = pd.DataFrame(comparison_table)
        csv_file = results_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)
    
    return {
        "processed_results": processed_file,
        "performance_summary": summary_file,
        "comparison_table": results_dir / "model_comparison.csv" if comparison_table else None
    }

def main():
    """Main results collection and processing function"""
    
    print("DUALPOI - RESULTS COLLECTION")
    print("=" * 50)
    print(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load raw results
        print("Loading raw experiment results...")
        raw_data = load_raw_results()
        total_experiments = raw_data.get("experiment_summary", {}).get("total_experiments", 0)
        print(f"Loaded results for {total_experiments} experiments")
        
        # Process results
        print("Processing experiment results...")
        processed_results = process_experiment_results(raw_data)
        
        # Create performance summary
        print("Creating performance summary...")
        performance_summary = create_performance_summary(processed_results)
        
        # Create comparison table
        print("Creating model comparison table...")
        comparison_table = create_comparison_table(processed_results)
        
        # Save all results
        print("Saving processed results...")
        saved_files = save_results(processed_results, performance_summary, comparison_table)
        
        print("RESULTS COLLECTION COMPLETED")
        print("=" * 50)
        print("Generated files:")
        for file_type, file_path in saved_files.items():
            if file_path:
                print(f"  {file_type}: {file_path}")
        
        print()
        print("PERFORMANCE SUMMARY:")
        if "error" not in performance_summary:
            print(f"  Best Accuracy: {performance_summary['accuracy']['max']:.3f} ({performance_summary['accuracy']['best_model']})")
            print(f"  Best Representativeness: {performance_summary['representativeness']['max']:.3f} ({performance_summary['representativeness']['best_model']})")
            print(f"  Average Accuracy: {performance_summary['accuracy']['mean']:.3f}")
            print(f"  Average Representativeness: {performance_summary['representativeness']['mean']:.3f}")
        else:
            print(f"  Error: {performance_summary['error']}")
        
    except Exception as e:
        print(f"ERROR: Results collection failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
