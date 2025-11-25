"""
Complete POI Recommender System Experiment Pipeline
Academic Implementation

This script executes all four model experiments in sequence:
1. Baseline Cluster Model
2. Baseline Hybrid Model  
3. Improved Cluster Model
4. Improved Hybrid Model

All experiments use real user features from FourSquare dataset
and generate representativeness analysis.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_experiment(script_name, experiment_name, script_dir):
    """Execute a single experiment and capture results"""
    print(f"Executing {experiment_name}...")
    
    start_time = time.time()
    original_dir = os.getcwd()
    
    try:
        script_path = project_root / script_dir / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS: {experiment_name} completed in {execution_time:.1f}s")
            return {
                "status": "success",
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"ERROR: {experiment_name} failed with return code {result.returncode}")
            return {
                "status": "error", 
                "execution_time": execution_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"TIMEOUT: {experiment_name} exceeded 30 minute limit")
        return {
            "status": "timeout",
            "execution_time": execution_time,
            "error": "Experiment exceeded time limit"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"EXCEPTION: {experiment_name} failed with exception: {str(e)}")
        return {
            "status": "exception",
            "execution_time": execution_time,
            "error": str(e)
        }
    finally:
        os.chdir(original_dir)

def extract_results_from_output(stdout_text):
    """Extract numerical results from experiment output"""
    results = {}
    
    try:
        lines = stdout_text.split('\n')
        
        for line in lines:
            # Extract accuracy
            if "Best Validation Accuracy:" in line:
                accuracy_str = line.split("Best Validation Accuracy:")[1].strip()
                accuracy_val = accuracy_str.split()[0].replace("(", "").replace(")", "")
                results["accuracy"] = float(accuracy_val)
            
            # Extract representativeness
            elif "OVERALL REPRESENTATIVENESS:" in line:
                repr_str = line.split("OVERALL REPRESENTATIVENESS:")[1].strip()
                repr_val = repr_str.split()[0]
                results["representativeness"] = float(repr_val)
            
            # Extract Hits@3 and Hits@5
            elif "Hits@3:" in line and "Hits@5:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "Hits@3:" in part:
                        results["hits_at_3"] = float(parts[i+1].replace(",", ""))
                    if "Hits@5:" in part:
                        results["hits_at_5"] = float(parts[i+1].replace(",", ""))
                        
    except Exception as e:
        print(f"Warning: Could not extract all results from output: {e}")
    
    return results

def main():
    """Execute all experiments and collect results"""
    
    print("POI RECOMMENDER SYSTEM - COMPLETE EXPERIMENT PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define experiments to run
    experiments = [
        {
            "script": "baseline_cluster_model.py",
            "name": "Baseline Cluster Model",
            "key": "baseline_cluster",
            "dir": "scripts"
        },
        {
            "script": "baseline_hybrid_model.py", 
            "name": "Baseline Hybrid Model",
            "key": "baseline_hybrid",
            "dir": "scripts"
        },
        {
            "script": "improved_cluster_experiment.py",
            "name": "Improved Cluster Model", 
            "key": "improved_cluster",
            "dir": "experiments"
        },
        {
            "script": "improved_hybrid_experiment.py",
            "name": "Improved Hybrid Model",
            "key": "improved_hybrid",
            "dir": "experiments"
        }
    ]
    
    # Execute all experiments
    all_results = {}
    total_start_time = time.time()
    
    for exp in experiments:
        print(f"Starting {exp['name']}...")
        
        # Run experiment
        result = run_experiment(exp["script"], exp["name"], exp["dir"])
        
        # Extract numerical results
        if result["status"] == "success":
            extracted_results = extract_results_from_output(result["stdout"])
            all_results[exp["key"]] = {
                "experiment_name": exp["name"],
                "status": "success",
                "execution_time": result["execution_time"],
                "results": extracted_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            all_results[exp["key"]] = {
                "experiment_name": exp["name"],
                "status": result["status"],
                "execution_time": result["execution_time"],
                "error": result.get("error", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            }
        
        print()
    
    total_execution_time = time.time() - total_start_time
    
    # Save results
    results_dir = project_root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "raw_experiment_results.json"
    
    final_results = {
        "experiment_summary": {
            "total_execution_time": total_execution_time,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "successful_experiments": sum(1 for r in all_results.values() if r["status"] == "success")
        },
        "experiments": all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("EXPERIMENT PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {total_execution_time:.1f}s")
    print(f"Results saved to: {results_file}")
    print()
    
    # Print summary
    successful = sum(1 for r in all_results.values() if r["status"] == "success")
    print(f"SUCCESSFUL EXPERIMENTS: {successful}/{len(experiments)}")
    
    for key, result in all_results.items():
        status = result["status"]
        if status == "success":
            results = result.get("results", {})
            accuracy = results.get("accuracy", "N/A")
            representativeness = results.get("representativeness", "N/A")
            print(f"  {result['experiment_name']}: Accuracy={accuracy}, Representativeness={representativeness}")
        else:
            print(f"  {result['experiment_name']}: FAILED ({status})")

if __name__ == "__main__":
    main()
