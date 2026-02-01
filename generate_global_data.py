import pandas as pd
import numpy as np

def generate_data():
    # Global grid
    lats = np.linspace(-80, 80, 50)
    lons = np.linspace(-180, 180, 100)
    times = np.arange(0, 5)
    
    data_list = []
    
    # Centers of blobs (Lat, Lon, Radius)
    centers = [
        (30, -60, 20),   # North Atlantic
        (-20, -100, 25), # South Pacific
        (10, 80, 15),    # Indian Ocean
        (50, 140, 20)    # North Pacific
    ]
    
    for t in times:
        for lat in lats:
            for lon in lons:
                # Base value is NaN
                val_temp = np.nan
                val_press = np.nan
                
                # Check if point is within any blob
                in_blob = False
                for clat, clon, rad in centers:
                    dist = np.sqrt((lat - clat)**2 + (lon - clon)**2)
                    if dist < rad:
                        # Generate a smooth gradient from center
                        # Move center slightly over time
                        offset = t * 2
                        dist_t = np.sqrt((lat - (clat+np.sin(t)*5))**2 + (lon - (clon+offset))**2)
                        
                        if dist_t < rad:
                            val_temp = 20 + 10 * np.exp(-dist_t**2 / (2 * (rad/2)**2))
                            val_press = 1000 + 50 * np.exp(-dist_t**2 / (2 * (rad/2)**2))
                            in_blob = True
                            break
                
                # We save all points to maintain grid structure, 
                # but "empty" points will have specific marker or NaN
                # For CSV simplicity, we just save the rows.
                # However, our visualizer expects a full grid.
                # So we save ALL points, but use NaN for empty.
                
                # To handle NaN in CSV: use empty string or a sentinel.
                # But let's use 0 for empty and filter later? 
                # Or better: Pandas handles NaN.
                
                # Optimization: To make "Patches" visible, we should set non-blob areas to NaN.
                
                row = {
                    'Time': t,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Temperature': val_temp if in_blob else np.nan,
                    'Pressure': val_press if in_blob else np.nan
                }
                data_list.append(row)

    df = pd.DataFrame(data_list)
    df.to_csv('new.csv', index=False)
    print("Generated new.csv with distributed data blobs.")

if __name__ == "__main__":
    generate_data()
