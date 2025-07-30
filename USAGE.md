# Usage Guide

## Basic Usage

### Running the Optimizer
```bash
python visualize.py
```

### Understanding the Output

#### Energy Efficiency Plots
The generated `ai_efficiency_plots.png` contains:
- **Top plot**: Energy consumption vs batch size for different precisions
- **Bottom plot**: Energy efficiency vs batch size
- **Optimal points**: Marked for each precision level

#### CSV Data Format
The `ai_energy_data.csv` contains:
- `batch_size`: Batch size tested (1-128)
- `fp32_energy`: Energy consumption for FP32 precision
- `fp16_energy`: Energy consumption for FP16 precision  
- `int8_energy`: Energy consumption for INT8 precision
- `fp32_efficiency`: Efficiency metric for FP32
- `fp16_efficiency`: Efficiency metric for FP16
- `int8_efficiency`: Efficiency metric for INT8

## Advanced Configuration

### Modifying Model Parameters
Edit the `ModelParams` struct in `ai_energy_lib.rs` to match your specific AI model characteristics.

### Custom Batch Size Ranges
Modify the export range in the main function:
```rust
optimizer.export_data(1, 256, "ai_energy_data.csv")?; // Test batch sizes 1-256
```

### Adding New Precision Types
Extend the precision matching in the energy calculation functions to support additional data types.