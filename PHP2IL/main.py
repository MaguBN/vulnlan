import argparse
import os
from .PHPDecomposer import PHPDecomposer
import sys

def arg_parser(args_list=None) -> tuple:
    parser = argparse.ArgumentParser(description='Translate the PHP program into a intermediate language (IL)')
    parser.add_argument('-s', nargs='*', help='Source :: Path to the PHP programs / directory')
    parser.add_argument('-d', nargs='?', default="../input_files/il/", help='Destination :: Path to the IL program / directory')
    args = parser.parse_args(args_list)

    for arg in args.s:
        if os.path.isfile(arg) and arg.endswith('.php'):
            continue
        elif os.path.isdir(arg):
            if not any(file.endswith('.php') for file in os.listdir(arg)):
                print(f"Error: The directory {arg} does not contain any PHP files.")
                return
        else:
            print(f"Error: {arg} is not a valid PHP file or directory.")
            return
    
    if not os.path.exists(args.d):
        args.d = "../input_files/il/"
        if not os.path.exists(args.d):
            os.makedirs(args.d)
            print(f"Directory {args.d} created.")

    args.d = os.path.abspath(args.d)
    if not args.d.endswith('/'):
        args.d = args.d + "/"

    return args.s, args.d


def php_parser(php_file) -> None:

    with open(php_file, 'r') as f:
        php_code = f.read()
    
        file_tokens = PHPDecomposer(php_code, file_id=php_file)
        tokens = file_tokens.get_php_tokens()
        
    return tokens


def il_file_writer(il_file_name, tokens, new_path) -> None:  

    if not os.path.exists(new_path):
        print("Folder /input_files/il/ for storing processed files does not exist.")
        return 
    
    il_file_name = os.path.basename(il_file_name)

    if os.path.exists(new_path + il_file_name[:-4] + ".il"):
        overwrite = str(input("File already exists. Do you wish to overwrite it? (Y/n) "))
        match overwrite.lower():
            case "y":
                pass
            case "yes":
                pass
            case "":
                pass
            case "n":
                return
            case _:
                print("Invalid input. File will not be overwritten.")
                return

    with open(new_path + il_file_name[:-4] + ".il", 'w') as f:
        for token in tokens:
            if token == "":
                continue
            f.write(token + "\n")
        print(f"File {il_file_name[:-4]}.il written in {new_path}.")
        return 1


def PHP2IL(args_list=None):
    source, dest = arg_parser(args_list)

    if not source:
        return
    
    php_files = []
    for arg in source:
        if os.path.isfile(arg):
            php_files.append(arg)
        elif os.path.isdir(arg):
            php_files += [os.path.join(arg, file) for file in os.listdir(arg) if file.endswith('.php')]
        else:
            print(f"Error: {arg} is not a valid PHP file or directory.")
            return


    for file in php_files:
        if file.endswith('.php'):
            file_tokens = php_parser(file)
            il_file_writer(file, file_tokens, dest)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # um teste descomplicado
        PHP2IL([
            '-s', 'data/raw/CWE_89__array-GET__func_FILTER-CLEANING-special_chars_filter__select_from-concatenation_simple_quote.php',
            '-d', '.'
        ])
    else:
        # a partir do terminal ou do launch json
        PHP2IL()