import os

# Get the current directory
current_directory = os.getcwd()

# Walk through the directory and rename subfolders
for root, dirs, files in os.walk(current_directory, topdown=False):
    for file_name in files:
        # Check if the folder name contains "K"
        if 'K' in file_name:
            old_path = os.path.join(root, file_name)
            new_file_name = file_name.replace('K', 'L')
            new_path = os.path.join(root, new_file_name)
            # Rename the folder
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')
