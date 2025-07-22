import os
import re

def remove_var_no_from_folder(input_dir, output_dir):
    

    # Check if folders exist
    if not os.path.exists(input_dir):
        print("Folder data/proc_func for storing processed files does not exist.")
        return 

    if not os.path.exists(output_dir):
        print("Folder data/proc_func_nono for storing processed files does not exist.")
        return 

    for file in os.listdir(input_dir):
        if file.endswith('.il'):
            file_path = os.path.join(input_dir, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                # Replace varXX or varXX_YY with var
                line = re.sub(r'\bvar\d+(?:_\d+)?\b', 'var', line)

                # Replace funcXX with func
                line = re.sub(r'\bfunction_name\d+\b', 'function_name', line)

                # Replace classXX with class
                line = re.sub(r'\bclass_name\d+\b', 'class_name', line)

                # Replace object->funcXX with object->func
                line = re.sub(r'->function_name\d+\b', '->function_name', line)

                new_lines.append(line)

            # Write processed lines to output file
            with open(os.path.join(output_dir, file), 'w') as f:
                f.writelines(new_lines)

    print("All files processed.")


if __name__ == '__main__':
    print("Processing files to remove variable numbers...")
    remove_var_no_from_folder("data/test_com_id/vuln", "data/test_sem_id/vuln")
    remove_var_no_from_folder("data/test_com_id/nvuln", "data/test_sem_id/nvuln")
    remove_var_no_from_folder("data/train_com_id/vuln", "data/train_sem_id/vuln")
    remove_var_no_from_folder("data/train_com_id/nvuln", "data/train_sem_id/nvuln")

