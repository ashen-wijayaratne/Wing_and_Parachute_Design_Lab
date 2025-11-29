import numpy as np
import matplotlib.pyplot as plt

# Constants
Cp_min, gamma = -0.81, 1.4  # Minimum Cp at cruise, Ratio of specific heats
M_start, M_end, dM = 0.3, 0.8, 0.025 # Mach sweep parameters
Mach = np.arange(M_start, M_end, dM)  # Mach number array

# Formulas
Cp_real = Cp_min / np.sqrt(1 - Mach**2)  # Compressibility-corrected Cp (Prandtl-Glauert) 
Cp_critical = (2 / (gamma * Mach**2)) * (((2 + (gamma - 1) * Mach**2) / (gamma + 1))**(gamma / (gamma - 1)) - 1) # Critical pressure coefficient (isentropic)

# Finding intersection (Cp_real = Cp_critical)
difference = Cp_real - Cp_critical
sign_change_indices = np.where(np.diff(np.sign(difference)))[0]

if len(sign_change_indices):
    i = sign_change_indices[0]
    M1, M2 = Mach[i], Mach[i+1]
    f1, f2 = difference[i], difference[i+1]
    critical_Mach = M1 - f1 * (M2 - M1) / (f2 - f1)  # Linear interpolation
    print(f"\n*** Critical Mach Found ***\nCritical Mach ≈ {critical_Mach:.7f}\n")
else:
    print("No intersection found in the given Mach range.")
    critical_Mach = None

# Plot
plt.figure(figsize=(10, 6))
plt.plot(Mach, Cp_real, label="Cp_real (compressibility corrected)", linewidth=2)
plt.plot(Mach, Cp_critical, label="Cp_critical (isentropic)", linewidth=2)

# Plot intersection point if found
if critical_Mach is not None:
    Cp_int = Cp_min / np.sqrt(1 - critical_Mach**2)
    plt.scatter([critical_Mach], [Cp_int], color='red', s=50, label="Intersection (Critical Mach)")

plt.xlabel("Mach Number")
plt.ylabel("Pressure Coefficient")
plt.title("Cp_real and Cp_critical vs Mach Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()