import os

# Get the current directory
current_directory = os.getcwd()

# Walk through the directory and rename subfolders
for root, dirs, files in os.walk(current_directory, topdown=False):
    for dir_name in dirs:
        # Check if the folder name contains "K"
        if 'K' in dir_name:
            old_path = os.path.join(root, dir_name)
            new_dir_name = dir_name.replace('K', 'L')
            new_path = os.path.join(root, new_dir_name)
            # Rename the folder
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')
