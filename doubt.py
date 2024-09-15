import pandas as pd
import os

# Load CSV data into a pandas DataFrame
csv_file = 'dataset/train.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Create output directory for separate CSV files
output_dir = 'entity_name_csv_files'
os.makedirs(output_dir, exist_ok=True)

# Group data by 'entity_name' and save each group to a separate CSV file
for entity_name, group in df.groupby('entity_name'):
    # Create a filename for each entity_name
    filename = f'{entity_name}.csv'
    
    # Save the group to a CSV file in the output directory
    group.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Saved {filename} with {len(group)} records.")

print("All files have been saved successfully.")
