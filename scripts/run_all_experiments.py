"""

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
            error_msg = f"Return code {result.returncode}"
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')[:10]
                error_msg = '; '.join(error_lines)
            elif result.stdout:
                error_lines = result.stdout.strip().split('\n')[-10:]
                error_msg = '; '.join(error_lines)
            
            print(f"ERROR: {experiment_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr[:500]}")
            if result.stdout:
                print(f"STDOUT (last 500 chars): {result.stdout[-500:]}")
            
            return {
                "status": "error", 
                "execution_time": execution_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": error_msg
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
        
        # Track best hits values (from final epoch or best epoch)
        best_hits_3 = 0.0
        best_hits_5 = 0.0
        import re
        
        for line in lines:
            # Extract accuracy - multiple formats
            if "Best Validation Accuracy:" in line:
                accuracy_str = line.split("Best Validation Accuracy:")[1].strip()
                accuracy_val = accuracy_str.split()[0].replace("(", "").replace(")", "")
                try:
                    results["accuracy"] = float(accuracy_val)
                except ValueError:
                    pass
            elif "Overall Accuracy:" in line or "Overall accuracy:" in line:
                # Format: "Overall Accuracy: 0.1932 (19.32%)"
                match = re.search(r'Overall Accuracy:\s*([\d.]+)\s*\(', line, re.IGNORECASE)
                if not match:
                    # Try without parentheses
                    match = re.search(r'Overall Accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    try:
                        acc_val = float(match.group(1))
                        # If > 1, it's percentage, otherwise decimal
                        results["accuracy"] = acc_val if acc_val < 1.0 else acc_val / 100.0
                    except ValueError:
                        pass
            elif re.search(r'Accuracy:\s*([\d.]+)%', line):
                # Format: "- Accuracy: 18.7%"
                match = re.search(r'Accuracy:\s*([\d.]+)%', line)
                if match:
                    try:
                        results["accuracy"] = float(match.group(1)) / 100.0
                    except ValueError:
                        pass
            
            # Extract representativeness - multiple formats
            if "OVERALL REPRESENTATIVENESS:" in line:
                repr_str = line.split("OVERALL REPRESENTATIVENESS:")[1].strip()
                repr_val = repr_str.split()[0]
                try:
                    results["representativeness"] = float(repr_val)
                except ValueError:
                    pass
            elif "Representativeness:" in line:
                # Format: "- Representativeness: 92.1%"
                match = re.search(r'Representativeness:\s*([\d.]+)%?', line)
                if match:
                    try:
                        repr_val = float(match.group(1))
                        # If > 1, assume it's percentage, otherwise assume it's already decimal
                        results["representativeness"] = repr_val / 100.0 if repr_val > 1.0 else repr_val
                    except ValueError:
                        pass
            
            # Extract Hits@3 and Hits@5 from epoch lines or final results
            if "Hits@3:" in line or "hits@3:" in line.lower():
                try:
                    hits3_match = re.search(r'Hits@3:\s*([\d.]+)', line, re.IGNORECASE)
                    if hits3_match:
                        hits3_val = float(hits3_match.group(1))
                        if hits3_val > best_hits_3:
                            best_hits_3 = hits3_val
                except (ValueError, AttributeError):
                    pass
            
            if "Hits@5:" in line or "hits@5:" in line.lower():
                try:
                    hits5_match = re.search(r'Hits@5:\s*([\d.]+)', line, re.IGNORECASE)
                    if hits5_match:
                        hits5_val = float(hits5_match.group(1))
                        if hits5_val > best_hits_5:
                            best_hits_5 = hits5_val
                except (ValueError, AttributeError):
                    pass
        
        # Use best values found (or last values if no best found)
        if best_hits_3 > 0:
            results["hits_at_3"] = best_hits_3
        if best_hits_5 > 0:
            results["hits_at_5"] = best_hits_5
                        
    except Exception as e:
        print(f"Warning: Could not extract all results from output: {e}")
    
    return results

def main():
    """Execute all experiments and collect results"""
    
    print("DUALPOI - COMPLETE EXPERIMENT PIPELINE")
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
            # Extract error message from stderr if available
            error_msg = result.get("error", "Unknown error")
            if not error_msg or error_msg == "Unknown error":
                stderr_text = result.get("stderr", "")
                stdout_text = result.get("stdout", "")
                if stderr_text:
                    # Take first few lines of stderr as error message
                    error_lines = stderr_text.strip().split('\n')[:5]
                    error_msg = '; '.join(error_lines) if error_lines else "Unknown error"
                elif stdout_text:
                    # If no stderr, check stdout for errors
                    error_lines = stdout_text.strip().split('\n')[-5:]
                    error_msg = '; '.join(error_lines) if error_lines else "Unknown error"
            
            all_results[exp["key"]] = {
                "experiment_name": exp["name"],
                "status": result["status"],
                "execution_time": result["execution_time"],
                "error": error_msg,
                "stdout": result.get("stdout", "")[:5000],  # Limit length
                "stderr": result.get("stderr", "")[:5000],  # Limit length
                "returncode": result.get("returncode"),
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
    
    # Always return True to allow pipeline to continue processing existing results
    # Pipeline will check stdout for "SUCCESSFUL EXPERIMENTS: 0/" to detect failures
    if successful == 0:
        print("\nWARNING: All experiments failed. Pipeline will continue to process existing results.")
    
    return True

if __name__ == "__main__":
    main()
