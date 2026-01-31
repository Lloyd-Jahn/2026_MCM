# Task 3: Water Resource Visualization Requirements

## Installation
```bash
pip install matplotlib seaborn numpy pandas
```

## File Structure
```
Task3/
└── Code/
    ├── main.py                           # Main orchestrator
    ├── sankey_water_cycle.py             # Sankey diagram module
    ├── violin_scenario_comparison.py     # Violin + box plot module
    ├── contour_sensitivity.py            # Contour + marginal distribution module
    ├── waterfall_cost.py                 # Waterfall chart module
    └── outputs/                          # Generated PDF outputs (auto-created)
        ├── sankey_water_cycle.pdf
        ├── violin_scenario_comparison.pdf
        ├── contour_sensitivity_analysis.pdf
        ├── waterfall_cost_breakdown.pdf
        └── Result.txt
```

## Usage
Run the main script from the Code directory:
```bash
cd Task3/Code
python main.py
```

## Outputs
All visualizations are saved as high-resolution PDFs (300 DPI) using absolute paths:
1. **sankey_water_cycle.pdf** - Water cycle balance diagram
2. **violin_scenario_comparison.pdf** - Scenario cost/time distributions
3. **contour_sensitivity_analysis.pdf** - Sensitivity analysis with marginal distributions
4. **waterfall_cost_breakdown.pdf** - Cost component breakdown for all scenarios
5. **Result.txt** - Statistical summary of findings

## Customization
Each module includes a "USER ADJUSTMENT GUIDE" in the header docstring:
- Color palettes (Hex codes)
- Font sizes (label_size, tick_size, legend_size)
- Figure dimensions
- Plot-specific parameters (e.g., contour levels, bar widths)

Line numbers are provided for easy editing.

## Design Principles
Following Nature-style guidelines:
- Times New Roman font, bold text
- Prominent black borders on all plots
- No internal titles (context via axis labels)
- Dynamic axis scaling to maximize data variance visibility
- High information density
- Professional color schemes
