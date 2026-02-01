import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

def process_sparse_data(input_file='raw_data.csv', output_file='dense_data.csv'):
    print(f"Processing {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # 1. Identify Columns
    # Assuming standard names from gen.py: 'Latitude', 'Longitude', 'Age'
    coords_cols = ['Latitude', 'Longitude', 'Age']
    var_cols = [c for c in df.columns if c not in coords_cols and c != 'SiteID']
    
    print(f"Variables to interpolate: {var_cols}")
    
    # 2. Define Target Grid
    # Create a regular grid for visualization
    # Lat: -90 to 90, Lon: -180 to 180, Age: min to max
    
    grid_lat_res = 50
    grid_lon_res = 100
    grid_time_res = 20
    
    lats = np.linspace(-90, 90, grid_lat_res)
    lons = np.linspace(-180, 180, grid_lon_res)
    ages = np.linspace(df['Age'].min(), df['Age'].max(), grid_time_res)
    
    # Create meshgrid (3D)
    # RBFInterpolator takes (N, D) points
    # We need to construct target points
    grid_lat_mesh, grid_lon_mesh, grid_age_mesh = np.meshgrid(lats, lons, ages, indexing='ij')
    
    # Flatten target grid
    target_points = np.column_stack([
        grid_lat_mesh.ravel(),
        grid_lon_mesh.ravel(),
        grid_age_mesh.ravel()
    ])
    
    # 3. Preprocessing / Normalization
    # Crucial: Normalize coordinates so distances are comparable
    # Scale all to [0, 1]
    
    def normalize(v, v_min, v_max):
        return (v - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(v)
    
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    min_age, max_age = df['Age'].min(), df['Age'].max()
    
    # 4. Interpolation Loop
    dense_data_dict = {
        'Latitude': grid_lat_mesh.ravel(),
        'Longitude': grid_lon_mesh.ravel(),
        'Time': grid_age_mesh.ravel() # Rename Age to Time for visualizer
    }
    
    for var in var_cols:
        print(f"Interpolating {var}...")
        
        # Filter NaN values for this variable
        sub_df = df.dropna(subset=[var] + coords_cols)
        
        if len(sub_df) < 10:
            print(f"  Warning: Not enough data points for {var}, skipping.")
            dense_data_dict[var] = np.nan
            continue
            
        # Source points
        src_lat = sub_df['Latitude'].values
        src_lon = sub_df['Longitude'].values
        src_age = sub_df['Age'].values
        src_vals = sub_df[var].values
        
        # Normalize Source
        src_lat_n = normalize(src_lat, min_lat, max_lat)
        src_lon_n = normalize(src_lon, min_lon, max_lon)
        src_age_n = normalize(src_age, min_age, max_age)
        
        src_points_n = np.column_stack([src_lat_n, src_lon_n, src_age_n])
        
        # Normalize Target
        tgt_lat_n = normalize(target_points[:,0], min_lat, max_lat)
        tgt_lon_n = normalize(target_points[:,1], min_lon, max_lon)
        tgt_age_n = normalize(target_points[:,2], min_age, max_age)
        
        tgt_points_n = np.column_stack([tgt_lat_n, tgt_lon_n, tgt_age_n])
        
        # Interpolate
        # kernel 'thin_plate_spline' is good for smooth geophysical fields
        # smoothing > 0 helps with noise
        try:
            rbf = RBFInterpolator(src_points_n, src_vals, kernel='thin_plate_spline', smoothing=0.1)
            interpolated_vals = rbf(tgt_points_n)
            
            # Optional: Mask values too far from any data point (extrapolation control)
            # Calculate distance to nearest source point for each target point?
            # That's expensive for large grids. 
            # Simple approach: Trust RBF but clip reasonable bounds.
            
            dense_data_dict[var] = interpolated_vals
            
        except Exception as e:
            print(f"  Error interpolating {var}: {e}")
            dense_data_dict[var] = np.nan

    # 5. Save Output
    dense_df = pd.DataFrame(dense_data_dict)
    
    # Sort for easier usage
    dense_df.sort_values(by=['Time', 'Latitude', 'Longitude'], inplace=True)
    
    dense_df.to_csv(output_file, index=False)
    print(f"Interpolation complete. Saved to {output_file}")
    print(f"Grid shape: {grid_lat_res}x{grid_lon_res}x{grid_time_res} = {len(dense_df)} rows")

if __name__ == "__main__":
    process_sparse_data()
