import os

def create_set_of_tokens(folder):
    if not os.path.exists(folder):
        print(f"Folder {folder} for storing processed files does not exist.")
        return 

    unique_tokens = []

    for file in os.listdir(folder):
        if file.endswith('.ann'):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                tokens = line.split()
                for token in tokens:
                    if token not in unique_tokens:
                        unique_tokens.append(token)

    with open(f"set_of_tokens_last.txt", 'w') as f:
        for token in unique_tokens:
            f.write(token + '\n')

    print(f"Set of unique tokens created and saved to set_of_tokens_last.txt.")

if __name__ == "__main__":
    create_set_of_tokens("hmm/nvuln_grupo_alto_last_ANN/")  # Adjust the folder path as needed