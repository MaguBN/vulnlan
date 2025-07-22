import os

def split_based_on_first_line():
    # Iterate over all files in the folder data/processed_var
    for file in os.listdir("data/proc_func_nono"):
        if file.endswith('.il'):
            file_path = os.path.join("data/proc_func_nono", file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Split based on the first line and create new files
            first_line = lines[0].strip()
            if '1' in first_line:
                new_file_path = os.path.join("data/vuln_nono", file)
            else:
                new_file_path = os.path.join("data/nvuln_nono", file)

            with open(new_file_path, 'w') as new_file:
                new_file.writelines(lines[1:])

    print("Files split based on the first line.")


if __name__ == '__main__':
    split_based_on_first_line()
        