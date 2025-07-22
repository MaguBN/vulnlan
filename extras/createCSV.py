import os
import csv

def read_files_and_create_dataset_from_il(directory, output_csv):
    dataset = []
    print(directory)
    for filename in os.listdir(directory):
        if filename.endswith(".il"):
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding="utf-8", mode='r') as file:
                lines = file.readlines()
                if lines:
                    vulnerable = 1 if "/vuln" in directory else 0
                    file_content = ''.join(lines[0:]).strip()
                    dataset.append([filename, file_content, vulnerable])

    with open(output_csv, encoding="utf-8", mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['file_name', 'file_content_in_il', 'vulnerable'])
        csvwriter.writerows(dataset)

def create_il_dataset(directory, output_csv):
    #directory = 'PHP2IL/data/processed_t'
    #output_csv = 'labelled_dataset_update_may.csv'
    read_files_and_create_dataset_from_il(directory, output_csv)
    return output_csv

def read_files_and_create_dataset_from_php(directory, output_csv):
    dataset = []

    for filename in os.listdir(directory):
        if filename.endswith(".php"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    vulnerable = lines[2].strip().split()
                    if vulnerable[0] == 'Safe':
                        vulnerable_n = 0
                    elif vulnerable[0] == 'Unsafe':
                        vulnerable_n = 1
                    #remove lines 2-7
                    lines = lines[:2] + lines[42:]
                
                    file_content = ''.join(lines[:]).strip()
                    dataset.append([filename, file_content, vulnerable_n])

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['file_name', 'file_content_in_php', 'vulnerable'])
        csvwriter.writerows(dataset)

def create_php_dataset(directory):
    directory = 'PHP2IL/data/raw'
    output_csv = 'labelled_dataset_php.csv'
    read_files_and_create_dataset_from_php(directory, output_csv)
    return output_csv

def read_files_and_create_dataset_from_php_test(directory, output_csv):
    dataset = []

    filename_cvs = "vuln_summary.cvs"
    if filename_cvs.endswith(".cvs"):
        file_path_cvs = os.path.join(directory, filename_cvs)
        with open(file_path_cvs, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                filename = line.split(",")[1]
                if line.split(",")[3] == 'unsafe':
                    vulnerable_n = 1
                elif line.split(",")[3] == 'safe':
                    vulnerable_n = 0
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    #remove lines 2-7
                    if "<!--" in lines[0]:
                        lines = lines[19:]
                    else:
                        lines = lines[:2] + lines[42:]
                
                    file_content = ''.join(lines[:]).strip()
                    dataset.append([filename, file_content, vulnerable_n])


    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['file_name', 'file_content_in_php', 'vulnerable'])
        csvwriter.writerows(dataset)

def create_php_dataset_test():
    directory = 'PHP2IL/data/All-cases'
    output_csv = 'labelled_dataset_php_test.csv'
    read_files_and_create_dataset_from_php_test(directory, output_csv)
    return output_csv

def main():
    #create_il_dataset('PHP2IL/data/proc_func', 'labelled_dataset_update_juneXX.csv')
    #create_il_dataset('PHP2IL/data/proc_func_nono', 'labelled_dataset_update_june.csv')
    
    create_il_dataset('data/test_sem_id/vuln', 'test_sem_id_il_vuln.csv')
    create_il_dataset('data/test_sem_id/nvuln', 'test_sem_id_il_nvuln.csv')
    create_il_dataset('data/train_sem_id/vuln', 'train_sem_id_il_vuln.csv')
    create_il_dataset('data/train_sem_id/nvuln', 'train_sem_id_il_nvuln.csv')

    #create_php_dataset_test()

if __name__ == '__main__':
    main()