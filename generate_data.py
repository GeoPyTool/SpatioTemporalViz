import numpy as np
import pandas as pd

def generate_data():
    # Settings
    n_lat = 40  # Increase resolution slightly
    n_lon = 40
    n_time = 100 # More time steps for smoother animation
    
    lats = np.linspace(30, 35, n_lat)
    lons = np.linspace(100, 105, n_lon)
    times = np.arange(n_time)
    
    # Create grid
    # Note: meshgrid default is xy indexing (cols, rows) -> (lon, lat)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    data = []
    
    print("Generating data...")
    for t in times:
        # Time factor for animation
        tf = t / 5.0
        
        # Temperature: wave moving diagonally + some noise
        # Base 25C, variation +/- 5
        temp = 25 + 5 * np.sin(lon_grid - tf) * np.cos(lat_grid - tf)
        
        # Pressure: Moving high pressure system
        # Center moves in a circle
        center_lon = 102.5 + 1.5 * np.sin(tf/2)
        center_lat = 32.5 + 1.5 * np.cos(tf/2)
        dist_sq = (lon_grid - center_lon)**2 + (lat_grid - center_lat)**2
        pressure = 1013 + 20 * np.exp(-dist_sq / 2.0)
        
        # pH: Generally stable around 7-8, influenced by "pollution" source
        # Source at (101, 31) pulsing
        dist_source = (lon_grid - 101)**2 + (lat_grid - 31)**2
        ph = 7.5 + 0.5 * np.sin(lon_grid*2 + lat_grid*2) - 1.0 * np.exp(-dist_source) * (0.5 + 0.5*np.sin(tf*2))
        
        # Oxygen: correlated with Temp (colder -> more oxygen) but also some biological consumption
        oxygen = 10 - (temp - 20) * 0.2 + 0.5 * np.random.normal(0, 0.1, temp.shape)
        
        # Flatten and append
        # We want a flat list to create DataFrame
        # Flattening arrays
        flat_lat = lat_grid.flatten()
        flat_lon = lon_grid.flatten()
        flat_temp = temp.flatten()
        flat_press = pressure.flatten()
        flat_ph = ph.flatten()
        flat_ox = oxygen.flatten()
        
        # Create a temporary dataframe for this timestep to speed up
        df_step = pd.DataFrame({
            'Time': t,
            'Latitude': flat_lat,
            'Longitude': flat_lon,
            'Temperature': flat_temp,
            'Pressure': flat_press,
            'pH': flat_ph,
            'Oxygen': flat_ox
        })
        data.append(df_step)
                
    final_df = pd.concat(data, ignore_index=True)
    
    # Save
    output_path = 'sample.csv'
    final_df.to_csv(output_path, index=False)
    print(f"{output_path} generated successfully with {len(final_df)} rows.")

if __name__ == "__main__":
    generate_data()
