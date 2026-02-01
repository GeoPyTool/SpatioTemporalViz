import numpy as np
import pandas as pd
import random

def generate_sparse_data():
    print("Generating sparse raw data...")
    
    # Configuration
    n_sites = 30           # Number of sampling sites
    n_samples_per_site = 10 # Max samples per site
    
    # 1. Generate Sites (Irregular locations)
    # Clusters of sites
    centers = [
        (40, -40),   # North Atlantic
        (-30, -110), # South Pacific
        (10, 70),    # Indian Ocean
        (60, 150)    # North Pacific
    ]
    
    site_coords = []
    for _ in range(n_sites):
        center = random.choice(centers)
        lat = center[0] + np.random.normal(0, 15)
        lon = center[1] + np.random.normal(0, 25)
        site_coords.append((lat, lon))
        
    data_rows = []
    
    # 2. Generate Data for each site
    for site_id, (lat, lon) in enumerate(site_coords):
        # Each site has data at random ages (Time)
        # e.g., geological cores or intermittent sensor logs
        n_obs = random.randint(3, n_samples_per_site)
        ages = np.sort(np.random.uniform(0, 10, n_obs)) # Time 0 to 10 Ma
        
        for age in ages:
            # Generate variables based on location and age
            # Add some noise and missing values
            
            # Base pattern
            temp = 25 - 0.2 * abs(lat) + 2 * np.sin(age) + np.random.normal(0, 1)
            pressure = 1000 + 0.1 * lat + 5 * np.cos(age) + np.random.normal(0, 5)
            ph = 8.1 - 0.01 * age + 0.05 * np.random.normal(0, 1)
            
            # Randomly drop some variables (sparse)
            if random.random() < 0.1: temp = np.nan
            if random.random() < 0.1: pressure = np.nan
            if random.random() < 0.1: ph = np.nan
            
            data_rows.append({
                'SiteID': f"S{site_id:02d}",
                'Latitude': lat,
                'Longitude': lon,
                'Age': age,
                'Temperature': temp,
                'Pressure': pressure,
                'pH': ph
            })
            
    df = pd.DataFrame(data_rows)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_file = 'raw_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} rows of sparse data in {output_file}")
    print(df.head())

if __name__ == "__main__":
    generate_sparse_data()
