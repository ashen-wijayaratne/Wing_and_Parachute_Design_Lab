import pandas as pd
import numpy as np

# User-configurable scoring and filter parameters
METRIC_WEIGHTS = {
    'L_D': 30,      # Efficiency (L/D ratio)
    'Cl_max': 25,   # Maximum lift coefficient
    'Cm': 20,       # Pitching moment stability (lower absolute value preferred)
    'Cd_min': 15,   # Minimum drag
    'Stall': 10     # Stall behavior (Gentle vs abrupt stall analysis for safety)
}
CM_FILTER_THRESHOLD = 0.1  # Maximum acceptable pitching moment (|Cm|)


# Function to analyze airfoils and compute performance metrics
def analyze_airfoils(csv_file):
    df = pd.read_csv(csv_file)
    
    # Dictionary to store results for each airfoil
    airfoil_results = {}
    print(f"Analyzing {len(df['Airfoil'].unique())} unique airfoils...")
    
    # Process each airfoil
    for airfoil in df['Airfoil'].unique():
        airfoil_data = df[df['Airfoil'] == airfoil].copy()
        
        # 1. Calculate (L/D)max - Weight: 30%
        airfoil_data['L_over_D'] = airfoil_data['CL'] / airfoil_data['CD']
        L_D_max = airfoil_data['L_over_D'].max()
        
        # 2. Maximum Lift Coefficient (Cl_max) - Weight: 25%
        Cl_max = airfoil_data['CL'].max()
        
        # 3. Pitching Moment Coefficient (Cm) - Weight: 20%
        # We want Cm close to zero (less negative is better)
        # Take average of absolute Cm values in linear range (alpha < 12°)
        linear_range = airfoil_data[airfoil_data['Alpha'] <= 12]
        avg_Cm = abs(linear_range['CM'].mean())
        
        # 4. Minimum Drag Coefficient (Cd_min) - Weight: 15%
        Cd_min = airfoil_data['CD'].min()
        
        # 5. Stall Characteristics - Weight: 10%
        # Gentle stall = CL doesn't drop sharply after stall
        # Calculate stall quality: CL at 20° / CL_max (higher is better - gentle stall)
        try:
            cl_at_20 = airfoil_data[airfoil_data['Alpha'] == 20]['CL'].values[0]
            stall_quality = cl_at_20 / Cl_max
        except:
            stall_quality = 0  # If no data at 20°, assume poor stall
        
        # Store results
        airfoil_results[airfoil] = {
            'L_D_max': L_D_max,
            'Cl_max': Cl_max,
            'avg_Cm': avg_Cm,
            'Cd_min': Cd_min,
            'stall_quality': stall_quality
        }
    
    return airfoil_results

# Function to calculate weighted scores for airfoils
def calculate_scores(airfoil_results):
    """Calculate weighted scores for each airfoil using user-configurable weights.

    Uses `METRIC_WEIGHTS` and `CM_FILTER_THRESHOLD` defined at module top so the
    weighting and filtering behavior can be changed without modifying core logic.
    """

    # Primary filter: Remove airfoils with |Cm| above user-configured threshold
    filtered_results = {airfoil: data for airfoil, data in airfoil_results.items() if data['avg_Cm'] <= CM_FILTER_THRESHOLD}
    print(f"Filtered out {len(airfoil_results) - len(filtered_results)} airfoils with |Cm| > {CM_FILTER_THRESHOLD}")

    # Gather metric lists for normalization
    L_D_max_values = [data['L_D_max'] for data in filtered_results.values()]
    Cl_max_values = [data['Cl_max'] for data in filtered_results.values()]
    avg_Cm_values = [data['avg_Cm'] for data in filtered_results.values()]
    Cd_min_values = [data['Cd_min'] for data in filtered_results.values()]
    stall_quality_values = [data['stall_quality'] for data in filtered_results.values()]

    # Safety defaults to avoid division-by-zero
    max_L_D_max = max(L_D_max_values) if L_D_max_values else 1.0
    max_Cl_max = max(Cl_max_values) if Cl_max_values else 1.0
    min_avg_Cm = min(avg_Cm_values) if avg_Cm_values else 1.0
    min_Cd_min = min(Cd_min_values) if Cd_min_values else 1.0
    max_stall_quality = max(stall_quality_values) if stall_quality_values else 1.0

    scored_airfoils = []

    # Calculate weighted scores for each filtered airfoil using METRIC_WEIGHTS
    for airfoil, data in filtered_results.items():
        # Efficiency (L/D)
        L_D_score = (data['L_D_max'] / max_L_D_max) * METRIC_WEIGHTS.get('L_D', 0)

        # Max lift (Cl_max)
        Cl_score = (data['Cl_max'] / max_Cl_max) * METRIC_WEIGHTS.get('Cl_max', 0)

        # Pitching moment: lower avg_Cm is better (inverse mapping)
        Cm_score = (min_avg_Cm / data['avg_Cm']) * METRIC_WEIGHTS.get('Cm', 0) if data['avg_Cm'] > 0 else METRIC_WEIGHTS.get('Cm', 0)

        # Minimum drag: lower Cd_min is better (inverse mapping)
        Cd_score = (min_Cd_min / data['Cd_min']) * METRIC_WEIGHTS.get('Cd_min', 0) if data['Cd_min'] > 0 else METRIC_WEIGHTS.get('Cd_min', 0)

        # Stall behavior: higher is better
        Stall_score = (data['stall_quality'] / max_stall_quality) * METRIC_WEIGHTS.get('Stall', 0)

        total_score = L_D_score + Cl_score + Cm_score + Cd_score + Stall_score

        scored_airfoils.append({
            'Airfoil': airfoil,
            'Total_Score': total_score,
            'L_D_max': data['L_D_max'],
            'Cl_max': data['Cl_max'],
            'avg_Cm': data['avg_Cm'],
            'Cd_min': data['Cd_min'],
            'stall_quality': data['stall_quality'],
            'Component_Scores': {
                'L_D': L_D_score,
                'Cl_max': Cl_score,
                'Cm': Cm_score,
                'Cd_min': Cd_score,
                'Stall': Stall_score
            }
        })

    # Sort descending
    scored_airfoils.sort(key=lambda x: x['Total_Score'], reverse=True)
    return scored_airfoils

# Function to print top airfoils
def main():
    csv_file = "xfoil_comprehensive_outputs/airfoil_data.csv"
    
    print("Analyzing airfoils with Cm FILTER (|Cm| <= 0.1)...")
    results = analyze_airfoils(csv_file)
    scored_airfoils = calculate_scores(results)
    
    print("\n" + "="*80)
    print("TOP AIRFOILS RANKED BY WEIGHTED MATRIX")
    print("FILTER: Only airfoils with |Cm| <= 0.1 considered")
    print("="*80)
    
    if not scored_airfoils:
        print("No airfoils passed the Cm filter! Try relaxing the criteria.")
        return
    
    top_count = min(10, len(scored_airfoils))
    
    for i, airfoil in enumerate(scored_airfoils[:top_count], 1):
        print(f"\n#{i}: {airfoil['Airfoil']}")
        print(f"   Total Score: {airfoil['Total_Score']:.2f}")
        print(f"   Component Scores:")
        print(f"     - L/D_max: {airfoil['Component_Scores']['L_D']:.2f} (value: {airfoil['L_D_max']:.2f})")
        print(f"     - Cl_max:  {airfoil['Component_Scores']['Cl_max']:.2f} (value: {airfoil['Cl_max']:.2f})")
        print(f"     - Cm:      {airfoil['Component_Scores']['Cm']:.2f} (avg |Cm|: {airfoil['avg_Cm']:.4f})")
        print(f"     - Cd_min:  {airfoil['Component_Scores']['Cd_min']:.2f} (value: {airfoil['Cd_min']:.5f})")
        print(f"     - Stall:   {airfoil['Component_Scores']['Stall']:.2f} (quality: {airfoil['stall_quality']:.3f})")
        
        # Print key performance metrics
        print(f"   Key Performance:")
        print(f"     - Max L/D: {airfoil['L_D_max']:.1f}")
        print(f"     - Max Cl:  {airfoil['Cl_max']:.2f}")
        print(f"     - Min Cd:  {airfoil['Cd_min']:.5f}")

if __name__ == "__main__":
    main()