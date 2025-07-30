import subprocess
import os
import sys

# Check for required dependencies
def check_dependencies():
    missing_packages = []
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_packages.append("matplotlib")
    
    if missing_packages:
        print("\nMissing required Python packages: " + ", ".join(missing_packages))
        print("\nPlease install the missing packages using one of these commands:")
        print("\nOption 1: Using pip (standard method):")
        print("pip install " + " ".join(missing_packages))
        print("\nOption 2: Using Python's executable:")
        print("python -m pip install " + " ".join(missing_packages))
        print("\nOption 3: If you're using Python 3 specifically:")
        print("pip3 install " + " ".join(missing_packages))
        print("\nAfter installing the packages, run this script again.")
        return False
    return True

def compile_and_run_rust():
    """Compile and run the Rust code to generate the CSV data"""
    print("Compiling Rust code...")
    result = subprocess.run(["rustc", "ai_energy_lib.rs"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation error:")
        print(result.stderr)
        return False
    
    print("Running Rust code to generate data...")
    result = subprocess.run(["ai_energy_lib.exe"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Runtime error:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def visualize_data():
    """Create visualizations from the CSV data"""
    # Import dependencies here to avoid errors if they're missing
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if not os.path.exists("ai_energy_data.csv"):
        print("Data file not found. Run the Rust code first.")
        return
    
    # Load the data
    data = pd.read_csv("ai_energy_data.csv")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Energy Consumption vs Batch Size
    ax1.plot(data['batch_size'], data['fp32_energy'], label='FP32')
    ax1.plot(data['batch_size'], data['fp16_energy'], label='FP16')
    ax1.plot(data['batch_size'], data['int8_energy'], label='INT8')
    ax1.set_title('Energy Consumption vs Batch Size')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Energy Consumption (joules)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Efficiency vs Batch Size
    ax2.plot(data['batch_size'], data['fp32_efficiency'], label='FP32')
    ax2.plot(data['batch_size'], data['fp16_efficiency'], label='FP16')
    ax2.plot(data['batch_size'], data['int8_efficiency'], label='INT8')
    ax2.set_title('Efficiency vs Batch Size')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Efficiency (samples/joule)')
    ax2.legend()
    ax2.grid(True)
    
    # Find optimal batch sizes
    for precision in ['fp32', 'fp16', 'int8']:
        energy_col = f'{precision}_energy'
        efficiency_col = f'{precision}_efficiency'
        
        # Find optimal batch size for efficiency
        optimal_idx = data[efficiency_col].idxmax()
        optimal_batch = data.loc[optimal_idx, 'batch_size']
        max_efficiency = data.loc[optimal_idx, efficiency_col]
        
        # Mark on the plot
        ax2.scatter(optimal_batch, max_efficiency, marker='o', s=100)
        ax2.annotate(f'{precision}: {int(optimal_batch)}', 
                    (optimal_batch, max_efficiency),
                    xytext=(10, 10), textcoords='offset points')
        
        print(f"Optimal batch size for {precision}: {int(optimal_batch)}")
        print(f"Maximum efficiency: {max_efficiency:.4f} samples/joule")
    
    plt.tight_layout()
    plt.savefig('ai_efficiency_plots.png')
    plt.show()
    
    print("Visualization saved to ai_efficiency_plots.png")

def main():
    print("AI Energy Efficiency Optimizer - Hybrid Approach")
    print("==============================================\n")
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Run Rust code to generate data
    if compile_and_run_rust():
        # Visualize the data
        visualize_data()

if __name__ == "__main__":
    main()