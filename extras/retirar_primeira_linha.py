import os

def retirar_primeira_linha(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        f.writelines(lines[1:])


def main(to_remove, folder=0):
    if folder==0: # when it is not a folder
        if os.path.isfile(to_remove):
            retirar_primeira_linha(to_remove)
            return 1
        else:
            print(f"File {to_remove} does not exist.")
            return 0
    else: # when it is a folder
        #simple code to get files .il from folder and remove first line
        for filename in os.listdir(to_remove):
            if filename.endswith('.il'):
                file = os.path.join(to_remove, filename)
                if os.path.isfile(file):
                    retirar_primeira_linha(file)
                    a = 1
                else:
                    print(f"File {file} does not exist.")
        if a == 1:
            return 1
        return 0

if __name__ == "__main__":
    folder = 'PHP2IL/hmm/test_vuln_grupo_alto_varXX'
    return_code = main(folder, 1)
    if return_code == 1:
        print("First line removed successfully and found files.")
    else:
        print("Failed to remove first line or didn't find files.")