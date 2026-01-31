"""
Sankey Diagram: Water Cycle Balance for Moon Colony
===================================================
Visualizes the flow of water from Earth transport → Colony usage → Recycling system

USER ADJUSTMENT GUIDE:
----------------------
Line 25-29:  Color palette (Hex codes)
Line 32:     Figure size
Line 56-70:  Flow widths and positions
Line 85-95:  Font sizes (label_size, annotation_size)
"""

import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Professional Color Palette (Nature-style)
COLOR_TRANSPORT = '#3498db'    # Blue - Transport from Earth
COLOR_RECYCLED = '#2ecc71'     # Green - Recycled water
COLOR_LOSS = '#e74c3c'         # Red - Water loss
COLOR_EMERGENCY = '#f39c12'    # Orange - Emergency reserve
COLOR_USAGE = '#95a5a6'        # Gray - Usage flow

# Plot Configuration
FIG_SIZE = (16, 10)
DPI = 300

# Font Configuration (Times New Roman, Bold)
LABEL_SIZE = 18
ANNOTATION_SIZE = 16
TICK_SIZE = 16

# ============================================================================
# SANKEY DIAGRAM CREATION
# ============================================================================

def create_sankey_diagram(water_data, output_path):
    """
    Create Sankey diagram showing water cycle balance

    Parameters:
    -----------
    water_data : dict
        Dictionary containing water demand and recycling data
    output_path : Path
        Absolute path to save the PDF
    """

    # Extract data
    daily_living = water_data['daily_living']
    production = water_data['production']
    emergency = water_data['emergency']
    recycling_rate = water_data['recycling_rate']
    net_daily = water_data['net_daily']
    net_emergency = water_data['net_emergency']
    total_transport = water_data['total_transport']

    # Calculate flow values (in thousand tons for better scale)
    # Convert to thousand tons
    total_usage = (daily_living + production) / 1e3      # 3102.5 thousand tons/year
    daily_living_kt = daily_living / 1e3                 # 1825 thousand tons/year
    production_kt = production / 1e3                     # 1277.5 thousand tons/year
    recycled_water = total_usage * recycling_rate        # 2947.375 thousand tons/year
    water_loss = total_usage * (1 - recycling_rate)      # 155.125 thousand tons/year
    transport_daily = net_daily / 1e3                    # 155.125 thousand tons/year
    transport_emergency = net_emergency / 1e3            # 150 thousand tons
    emergency_kt = emergency / 1e3                       # 150 thousand tons

    # Create figure
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    # Create Sankey diagram with simpler 3-node structure
    sankey = Sankey(ax=ax, scale=0.0003, offset=0.5, head_angle=120,
                    format='%.1f', unit=' kt/year',
                    shoulder=0.05, gap=0.6, tolerance=1e-3)

    # Node 1: Earth Transport (input from Earth)
    sankey.add(flows=[transport_daily, transport_emergency, -(transport_daily + transport_emergency)],
               labels=['Daily Transport\n(155 kt/year)', 'Emergency Reserve\n(150 kt)', 'To Colony\nReservoir'],
               orientations=[0, 0, 0],
               pathlengths=[0.3, 0.3, 0.2],
               facecolor=COLOR_TRANSPORT,
               edgecolor='black',
               linewidth=1,
               alpha=0.85)

    # Node 2: Colony Usage (daily living + production ONLY, emergency stored separately)
    # Balance: input = output
    # Input: transport_daily (155) + recycled_water (2947.375) = 3102.375
    # Output: daily_living (1825) + production (1277.5) = 3102.5
    # They must be equal! Use actual consumption values.
    sankey.add(flows=[transport_daily + transport_emergency, recycled_water,
                      -(daily_living_kt + production_kt), -emergency_kt],
               labels=['From Earth', 'Recycled Water\n(2947 kt/year)',
                      'Daily Usage\n(3103 kt/year)', 'Emergency Storage\n(150 kt)'],
               orientations=[0, -1, 1, 0],
               pathlengths=[0.2, 0.5, 0.3, 0.3],
               prior=0,
               connect=(2, 0),
               facecolor=COLOR_USAGE,
               edgecolor='black',
               linewidth=1,
               alpha=0.85)

    # Node 3: Recycling System (95% efficiency)
    # Balance: input = output
    # Input: daily_living + production = 3102.5
    # Output: recycled_water (2947.375) + water_loss (155.125) = 3102.5 ✓
    sankey.add(flows=[daily_living_kt + production_kt, -recycled_water, -water_loss],
               labels=['Used Water\n(3103 kt/year)', 'Back to Colony\n(95% recycled)',
                      'Water Loss\n(5%, 155 kt/year)'],
               orientations=[0, -1, 0],
               pathlengths=[0.3, 0.5, 0.3],
               prior=1,
               connect=(2, 0),
               facecolor=COLOR_RECYCLED,
               edgecolor='black',
               linewidth=1,
               alpha=0.85)

    # Render the diagram
    diagrams = sankey.finish()

    # Customize appearance
    for diagram in diagrams:
        for text in diagram.texts:
            text.set_fontname('Times New Roman')
            text.set_fontsize(ANNOTATION_SIZE)
            text.set_fontweight('bold')

    # Add title via xlabel (no internal title per guideline)
    ax.set_xlabel('Water Flow from Earth Transport → Colony Usage → Recycling System',
                  fontsize=LABEL_SIZE, fontweight='bold', fontname='Times New Roman')

    # Remove axes
    ax.axis('off')

    # Add text annotations for key metrics
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2)

    ax.text(0.02, 0.98, f'Total Transport Required: {total_transport/1e5:.2f} × 10⁵ tons',
            transform=ax.transAxes, fontsize=ANNOTATION_SIZE, fontweight='bold',
            fontname='Times New Roman', verticalalignment='top', bbox=bbox_props)

    ax.text(0.02, 0.90, f'Recycling Rate: {recycling_rate*100:.0f}%',
            transform=ax.transAxes, fontsize=ANNOTATION_SIZE, fontweight='bold',
            fontname='Times New Roman', verticalalignment='top', bbox=bbox_props)

    ax.text(0.02, 0.82, f'Net Reduction: {(1-recycling_rate)*100:.0f}% → 5% of total demand',
            transform=ax.transAxes, fontsize=ANNOTATION_SIZE, fontweight='bold',
            fontname='Times New Roman', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', edgecolor='black', linewidth=2, alpha=0.3))

    # Tight layout
    plt.tight_layout()

    # Save as high-resolution PDF
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Sankey diagram saved: {output_path}")


if __name__ == '__main__':
    # Test with sample data
    from pathlib import Path

    water_data = {
        'daily_living': 1.825e6,
        'production': 1.2775e6,
        'emergency': 1.5e5,
        'recycling_rate': 0.95,
        'net_daily': 1.55125e5,
        'net_emergency': 1.5e5,
        'total_transport': 3.05125e5
    }

    output_path = Path(__file__).parent / 'outputs' / 'test_sankey.pdf'
    output_path.parent.mkdir(exist_ok=True)
    create_sankey_diagram(water_data, output_path)
    print(f"Test completed! Check {output_path}")
