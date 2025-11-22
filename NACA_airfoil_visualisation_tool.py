import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# User Configurable NACA Parameters
m = 0.03    # Maximum camber (% of chord)
p = 0.2     # Position of maximum camber (% of chord)
t = 0.13    # Maximum thickness (% of chord)
c = 1.0     # Chord length (normalized)

# Generate x-coordinates using cosine spacing for better resolution near leading edge
n_points = 200
beta = np.linspace(0, np.pi, n_points)
x = (1 - np.cos(beta)) / 2 * c

# Calculate thickness distribution (NACA 4-digit formula)
yt = 5 * t * c * (
    0.2969 * np.sqrt(x/c) - 
    0.1260 * (x/c) - 
    0.3516 * (x/c)**2 + 
    0.2843 * (x/c)**3 - 
    0.1015 * (x/c)**4
)

# Calculate mean camber line and its derivative
yc = np.zeros_like(x)
dyc_dx = np.zeros_like(x)

for i, xi in enumerate(x):
    xi_c = xi / c
    if xi_c < p:
        yc[i] = (m / p**2) * (2*p*xi_c - xi_c**2) * c
        dyc_dx[i] = (2*m / p**2) * (p - xi_c)
    else:
        yc[i] = (m / (1-p)**2) * ((1 - 2*p) + 2*p*xi_c - xi_c**2) * c
        dyc_dx[i] = (2*m / (1-p)**2) * (p - xi_c)

# Calculate surface coordinates
theta = np.arctan(dyc_dx)
xu = x - yt * np.sin(theta)
yu = yc + yt * np.cos(theta)
xl = x + yt * np.sin(theta)
yl = yc - yt * np.cos(theta)

# Find maximum camber position
imax_camber = np.argmax(yc)
x_max_camber = x[imax_camber]
y_max_camber = yc[imax_camber]

# Find maximum thickness position
imax_thickness = np.argmax(2*yt)
x_max_thickness = x[imax_thickness]
y_top_thickness = yc[imax_thickness] + yt[imax_thickness]
y_bot_thickness = yc[imax_thickness] - yt[imax_thickness]
t_max = y_top_thickness - y_bot_thickness

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(16, 6), dpi=120)

# Plot airfoil surfaces with distinct colors
ax.plot(xu, yu, 'b-', linewidth=2.5, label='Upper surface', zorder=3)
ax.plot(xl, yl, 'r-', linewidth=2.5, label='Lower surface', zorder=3)

# Plot mean camber line
ax.plot(x, yc, 'g--', linewidth=2, label='Mean camber line', zorder=2)

# Plot chord line
ax.plot([0, c], [0, 0], 'k:', linewidth=1.8, label='Chord line', zorder=1)

# Mark leading and trailing edges
ax.plot([0, c], [0, 0], 'ko', markersize=8, zorder=5)
ax.text(0, -0.045*c, "Leading Edge\n(0, 0)", ha="center", va="top", 
        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
ax.text(c, -0.045*c, "Trailing Edge\n(1.0c, 0)", ha="center", va="top", 
        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Mark maximum camber position
ax.plot(x_max_camber, y_max_camber, 'go', markersize=10, zorder=6)
ax.annotate(f"Maximum Camber\nx = {x_max_camber:.4f}c\ny = {y_max_camber:.4f}c\n(at {p*100:.0f}% chord)",
            xy=(x_max_camber, y_max_camber),
            xytext=(x_max_camber + 0.35*c, y_max_camber + 0.07*c),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                     edgecolor='green', alpha=0.9))

# Mark maximum thickness position
ax.plot(x_max_thickness, (y_top_thickness + y_bot_thickness)/2, 'mo', 
        markersize=10, zorder=6)

# Draw thickness measurement line
ax.plot([x_max_thickness, x_max_thickness], [y_bot_thickness, y_top_thickness], 
        'm-', linewidth=2, zorder=4)
ax.plot([x_max_thickness, x_max_thickness], [y_bot_thickness, y_top_thickness], 
        'mo', markersize=6, zorder=5)

ax.annotate(f"Maximum Thickness\nx = {x_max_thickness:.4f}c\nt = {t_max:.4f}c ({t*100:.0f}% chord)",
            xy=(x_max_thickness, y_top_thickness),
            xytext=(x_max_thickness - 0.15*c, y_top_thickness + 0.10*c),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='magenta'),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', 
                     edgecolor='magenta', alpha=0.9))

# Add thickness dimension label
ax.text(x_max_thickness + 0.015*c, (y_top_thickness + y_bot_thickness)/2, 
        f"{t_max:.4f}c", fontsize=9, va="center", ha="left",
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Add title and specifications
title_text = f"NACA {int(m*100)}{int(p*10)}{int(t*100)} Airfoil — Technical Diagram"
ax.text(0.70*c, 0.20*c, title_text, fontsize=16, fontweight='bold', ha='left')

specs_text = (f"Specifications:\n"
              f"• Maximum camber: m = {m:.2f}c ({m*100:.0f}% chord)\n"
              f"• Camber position: p = {p:.1f}c ({p*100:.0f}% chord)\n"
              f"• Maximum thickness: t = {t:.2f}c ({t*100:.0f}% chord)")
ax.text(0.70*c, 0.17*c, specs_text, fontsize=9, va='top', ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                 edgecolor='gray', alpha=0.9))

# Add legend
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

# Set axis properties
ax.set_xlim(-0.15*c, 1.20*c)
ax.set_ylim(-0.25*c, 0.25*c)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_xlabel('x/c (Chordwise position)', fontsize=11)
ax.set_ylabel('y/c (Vertical position)', fontsize=11)

# Add subtle background
ax.set_facecolor('#f8f8f8')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save with proper path handling
output_dir = Path(r"C:\Users\Ashen Wijayaratne\Code Projects\Python_Projects\Aero_report3")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "NACA_3213_technical_diagram.png"

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Diagram saved to: {output_path}")
plt.show()