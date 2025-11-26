#!/usr/bin/env python3
"""
Run Jupyter Notebooks Programmatically
Converts and executes notebooks, handling errors
"""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_notebook_with_nbconvert(notebook_path, timeout=600):
    """Run notebook using nbconvert with execution"""
    
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    print(f"Running notebook: {notebook_path.name}")
    print("=" * 60)
    
    # Create output directory
    output_dir = project_root / "results" / "notebook_executions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run notebook using nbconvert
    output_file = output_dir / f"{notebook_path.stem}_executed.ipynb"
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
                "--output-dir", str(output_dir),
                "--ExecutePreprocessor.timeout", str(timeout),
                "--ExecutePreprocessor.kernel_name", "python3"
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout + 60
        )
        
        if result.returncode == 0:
            print(f"Notebook executed successfully!")
            print(f"   Output saved to: {output_file}")
            return True, result.stdout, result.stderr
        else:
            print(f"  Notebook execution completed with warnings/errors")
            print(f"   STDOUT: {result.stdout[:500]}")
            print(f"   STDERR: {result.stderr[:500]}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f" Notebook execution timeout after {timeout} seconds")
        return False, "", "Timeout exceeded"
    except Exception as e:
        print(f" Error executing notebook: {str(e)}")
        return False, "", str(e)

def run_notebook_as_script(notebook_path):
    """Convert notebook to Python script and execute"""
    
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    print(f"Converting and running notebook: {notebook_path.name}")
    print("=" * 60)
    
    # Convert to Python script
    script_path = project_root / "results" / "notebook_executions" / f"{notebook_path.stem}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert notebook to Python
        convert_result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "python",
                str(notebook_path),
                "--output", str(script_path.stem),
                "--output-dir", str(script_path.parent)
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if convert_result.returncode != 0:
            print(f" Failed to convert notebook: {convert_result.stderr}")
            return False, "", convert_result.stderr
        
        print(f" Notebook converted to: {script_path}")
        
        # Execute the script
        print(f"Executing script...")
        exec_result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if exec_result.returncode == 0:
            print(f" Script executed successfully!")
            print(f"   Output length: {len(exec_result.stdout)} characters")
            return True, exec_result.stdout, exec_result.stderr
        else:
            print(f"  Script execution completed with errors")
            print(f"   STDOUT: {exec_result.stdout[:500]}")
            print(f"   STDERR: {exec_result.stderr[:500]}")
            return False, exec_result.stdout, exec_result.stderr
            
    except subprocess.TimeoutExpired:
        print(f" Script execution timeout")
        return False, "", "Timeout exceeded"
    except Exception as e:
        print(f" Error: {str(e)}")
        return False, "", str(e)

def main():
    """Main function to run notebooks"""
    
    print("NOTEBOOK EXECUTION TOOL")
    print("=" * 60)
    print()
    
    notebooks_dir = project_root / "notebooks"
    notebooks = [
        notebooks_dir / "recsys_knowledge_graph.ipynb",
        notebooks_dir / "frsqr_2025.ipynb"
    ]
    
    results = {}
    
    for notebook_path in notebooks:
        if not notebook_path.exists():
            print(f"  Skipping {notebook_path.name}: not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {notebook_path.name}")
        print(f"{'='*60}\n")
        
        # For knowledge graph notebook, we have a dedicated script
        if "recsys_knowledge_graph" in notebook_path.name:
            print("ℹ️  Using dedicated script: generate_knowledge_graph_visualizations.py")
            print("   (This script already implements the visualization functionality)")
            
            try:
                from scripts.generate_knowledge_graph_visualizations import main as generate_viz
                generate_viz()
                results[notebook_path.name] = {"status": "success", "method": "dedicated_script"}
                print(" Knowledge graph visualizations generated successfully!")
            except Exception as e:
                print(f"  Error: {e}")
                results[notebook_path.name] = {"status": "error", "error": str(e)}
            
            continue
        
        # For other notebooks, try to run as script
        print("Converting notebook to Python script...")
        success, stdout, stderr = run_notebook_as_script(notebook_path)
        
        results[notebook_path.name] = {
            "status": "success" if success else "error",
            "stdout_length": len(stdout),
            "stderr_length": len(stderr)
        }
        
        print()
    
    # Summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    
    for nb_name, result in results.items():
        status_text = "[OK]" if result["status"] == "success" else "[FAIL]"
        print(f"{status_text} {nb_name}: {result['status']}")
    
    print()
    print("Note: Some notebooks may require manual execution in Jupyter")
    print("      for full interactivity (especially visualization cells)")

if __name__ == "__main__":
    main()

