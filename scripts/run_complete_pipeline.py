"""
Complete POI Recommender System Pipeline


This script executes the complete experimental pipeline:
1. Run all model experiments
2. Collect and process results
3. Generate visualizations

Provides automation for reproducible research.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_script(script_name, description):
    """Execute a Python script and return results"""
    print(f"Executing: {description}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        script_path = project_root / "scripts" / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed in {execution_time:.1f}s")
            return True, result.stdout, result.stderr
        else:
            print(f"ERROR: {description} failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"TIMEOUT: {description} exceeded 1 hour limit")
        return False, "", "Timeout exceeded"
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"EXCEPTION: {description} failed: {str(e)}")
        return False, "", str(e)

def check_prerequisites():
    """Check if all required files and directories exist"""
    print("Checking prerequisites...")
    
    required_files = [
        "scripts/run_all_experiments.py",
        "scripts/collect_all_results.py", 
        "scripts/generate_all_visualizations.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        return False
    
    # Check if results directory exists, create if not
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    (results_dir / "metrics").mkdir(exist_ok=True)
    (results_dir / "visualizations").mkdir(exist_ok=True)
    
    print("All prerequisites satisfied")
    return True

def create_pipeline_summary(success_steps, total_time):
    """Create summary of pipeline execution"""
    
    summary = f"""
COMPLETE PIPELINE EXECUTION SUMMARY
===================================
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Execution Time: {total_time:.1f} seconds
Successful Steps: {len(success_steps)}/3

EXECUTION RESULTS:
"""
    
    steps = [
        "Model Experiments",
        "Results Collection", 
        "Visualization Generation"
    ]
    
    for i, step in enumerate(steps, 1):
        if step in success_steps:
            summary += f"  {i}. {step}: SUCCESS\n"
        else:
            summary += f"  {i}. {step}: NOT EXECUTED\n"
    
    return summary

def main():
    """Main pipeline execution function"""
    
    print("POI RECOMMENDER SYSTEM - COMPLETE PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    pipeline_start_time = time.time()
    successful_steps = []
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            print("Pipeline aborted due to missing prerequisites")
            sys.exit(1)
        
        print()
        
        # Step 1: Run all experiments
        success, stdout, stderr = run_script(
            "run_all_experiments.py",
            "Running all model experiments"
        )
        
        if success:
            successful_steps.append("Model Experiments")
        else:
            print("Pipeline stopped: Experiments failed")
            print(create_pipeline_summary(successful_steps, time.time() - pipeline_start_time))
            sys.exit(1)
        
        print()
        
        # Step 2: Collect results
        success, stdout, stderr = run_script(
            "collect_all_results.py", 
            "Collecting and processing results"
        )
        
        if success:
            successful_steps.append("Results Collection")
        else:
            print("Pipeline stopped: Results collection failed")
            print(create_pipeline_summary(successful_steps, time.time() - pipeline_start_time))
            sys.exit(1)
        
        print()
        
        # Step 3: Generate visualizations
        success, stdout, stderr = run_script(
            "generate_all_visualizations.py",
            "Generating visualizations"
        )
        
        if success:
            successful_steps.append("Visualization Generation")
        else:
            print("Warning: Visualization generation failed, continuing...")
        
        print()
        
        # Pipeline completion
        total_time = time.time() - pipeline_start_time
        
        print("COMPLETE PIPELINE FINISHED")
        print("=" * 60)
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Successful steps: {len(successful_steps)}/3")
        
        # Print summary
        summary = create_pipeline_summary(successful_steps, total_time)
        print(summary)
        
        # Save summary to file
        summary_file = project_root / "results" / "pipeline_execution_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"Pipeline summary saved to: {summary_file}")
        
        if len(successful_steps) >= 2:
            print("Pipeline completed successfully!")
        else:
            print("Pipeline completed with errors")
            sys.exit(1)
        
    except KeyboardInterrupt:
        total_time = time.time() - pipeline_start_time
        print(f"\nPipeline interrupted by user after {total_time:.1f}s")
        sys.exit(1)
    except Exception as e:
        total_time = time.time() - pipeline_start_time
        print(f"\nPipeline failed with exception: {str(e)}")
        print(create_pipeline_summary(successful_steps, total_time))
        sys.exit(1)

if __name__ == "__main__":
    main()
