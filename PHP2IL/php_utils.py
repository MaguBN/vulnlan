import re
from nltk.tokenize import regexp_tokenize

def remove_comments(code):
    code = remove_inline_comments(code)
    code = remove_multiline_comments(code)
    return code

def remove_inline_comments(code):
    return re.sub(r"(//.*?$)|(#.*?$)", '', code, flags=re.MULTILINE)

def remove_multiline_comments(code):
    return re.sub(r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/", '', code)

def remove_heredoc(code):
    # Reminder - Heredoc can expand variables (XSS)
    
    def process_heredoc(match):
        heredoc_content = match.group(1)
        # Remove static text while preserving variables
        processed_content = re.sub(r'[^$]*', '', heredoc_content)
        return f'<<<EOD\n{processed_content}\nEOD;'
    
    return re.sub(r'<<<(\w+)(.*?)\1;', process_heredoc, code, flags=re.DOTALL)

def remove_nowdoc(code):
    return re.sub(r'<<<\'(\w+)\'(.*?)\1;', '', code, flags=re.DOTALL)

def get_php_snippets(code):
    return re.findall(r'<\?php(.*?)\?>', code, re.DOTALL)

def obfuscate_variables(lines):
    mapping = {}
    counter = [1]  # Usamos uma lista para modificar o valor dentro da função interna

    def replace_variable(match):
        var_name = match.group(1)
        if var_name not in mapping:
            mapping[var_name] = f"var{counter[0]}"
            counter[0] += 1
        return mapping[var_name]
    
    return [re.sub(r'\$(\w+)', replace_variable, line) for line in lines]

def preserve_indentation(lines, func):
    # Guarda as indentações originais de cada linha
    indents = [len(line) - len(line.lstrip(' ')) for line in lines]
    
    # Remove indentação para enviar para a função de processamento
    stripped_lines = [line.lstrip(' ') for line in lines]
    
    # Chama a função passando o texto completo (lista de linhas sem indentação)
    processed_lines = func(stripped_lines)
    
    # Reaplica a indentação original linha a linha
    new_lines = []
    for indent, pline in zip(indents, processed_lines):
        new_lines.append(' ' * indent + pline)
    
    return new_lines

def obfuscate_classes_and_functions(lines):
    class_map = {}
    func_map = {}
    class_counter = 1
    func_counter = 1

    # 1. Detectar classes e funções
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("class "):
            # ex: class Input
            cls = line_strip.split()[1]
            if cls not in class_map:
                class_map[cls] = f"class_name{class_counter}"
                class_counter += 1
        elif "function" in line_strip:
            tokens = line_strip.split()
            if "function" in tokens:
                idx = tokens.index("function")
                if idx + 1 < len(tokens):
                    func = tokens[idx + 1]
                    if func not in func_map:
                        func_map[func] = f"function_name{func_counter}"
                        func_counter += 1

    updated_lines = []
    for line in lines:
        # manter espaços iniciais para preservar o formato
        leading_spaces = len(line) - len(line.lstrip(' '))

        # para facilitar a substituição, trabalhar com tokens simples
        tokens = line.strip().split()

        if not tokens:
            # linha vazia, só manter
            updated_lines.append(line)
            continue

        # 2. Substituir declaração de classe
        if tokens[0] == "class" and len(tokens) > 1:
            tokens[1] = class_map.get(tokens[1], tokens[1])

        # 3. Substituir declaração de função
        if "function" in tokens:
            idx = tokens.index("function")
            if idx + 1 < len(tokens):
                tokens[idx + 1] = func_map.get(tokens[idx + 1], tokens[idx + 1])

        # 4. Substituir new Class
        if "new" in tokens:
            idx = tokens.index("new")
            if idx + 1 < len(tokens):
                tokens[idx + 1] = class_map.get(tokens[idx + 1], tokens[idx + 1])

        # 5. Substituir chamadas método var->method
        for i, token in enumerate(tokens):
            if '->' in token:
                parts = token.split('->')
                if len(parts) == 2:
                    obj, method = parts
                    method_new = func_map.get(method, method)
                    tokens[i] = f"{obj}->{method_new}"

        # reconstruir linha com os espaços originais
        new_line = (' ' * leading_spaces) + " ".join(tokens)
        updated_lines.append(new_line)

    return updated_lines

def join_obs_index(lines):
    for idx, line in enumerate(lines):
        tokens = line.split()  # Dividir a linha em tokens
        last_var = None  # Variável para manter a última variável encontrada

        # Processar tokens
        for i in range(len(tokens)):
            # Verificar se o token atual é uma variável e manter a última variável encontrada
            if tokens[i].startswith("var"):
                last_var = tokens[i]

            # Verificar se o token atual é um número e há uma variável anterior
            elif re.search(r'\d+', tokens[i]) and last_var and tokens[i] != "4096":
                # Juntar a última variável com o número
                tokens[i-1] = last_var + "_" + tokens[i]
                tokens[i] = ""  # Limpar o token numérico
                last_var = None  # Resetar a variável para evitar repetições

        # Reconstruir a linha a partir dos tokens e limpar os espaços extras
        new_line = " ".join(tokens).strip()
        lines[idx] = re.sub(r'\s+', ' ', new_line)  # Remover múltiplos espaços

    return lines

def process_conditional_statement(line, indentation_count): #, dict
    line = obtain_query(line)
    line = remove_strings(line)

    if_pattern = r'\w+|\$[\w_]+|<|>|==|<=|>=|&&|"[^"]*"'
    logic_operators_pattern = r'\b(?:and|&&|or|\|\||xor)\b'
    string_pattern = r'(\$[a-zA-Z_]\w*)\s*==\s*".*?"|".*?"\s*==\s*(\$[a-zA-Z_]\w*)'

    new_line = regexp_tokenize(line.strip(), if_pattern)
    if new_line[0] == "else":
        return " " * (indentation_count * 2) + ' '.join(new_line)
    
    #process if it is a elseif or if
    processed_line = (' '.join(new_line))[len(new_line[0]):]
    split_line = re.split(logic_operators_pattern, processed_line, flags=re.IGNORECASE)
    result = new_line[0] + " "
    for s in split_line:
        match = re.search(string_pattern, s)
        if match:
            var = match.group(1) or match.group(2)
            result += var + " string_value and "
        else:
            result += process_normal_line(s, indentation_count).strip() + " and " #, dict
    return " " * (indentation_count * 2) + result[:-4]

def remove_superglobal_prefix(code):
    # Lista dos superglobais do PHP
    superglobals = ["$_GET", "$_POST", "$_REQUEST", "$_SESSION", "$_COOKIE", "$_FILES", "$_ENV", "$_SERVER", "$_GLOBALS"]
    
    for superglobal in superglobals:
        # Substitui o superglobal com $_ pela versão sem o prefixo
        code = code.replace(superglobal, superglobal[2:])
    
    return code

def obtain_query(line):
    sql_tokens = ["SELECT", "UPDATE", "DELETE", "INSERT", "CREATE", "ALTER", "DROP", "SET"]
    
    # Criar regex para tokens SQL seguidos de espaço ou final de linha
    sql_pattern = r'(?<![\w])(?:' + '|'.join(sql_tokens) + r')(?=\s|$)'

    # Verifica se contém algum token SQL
    match = re.search(sql_pattern, line, re.IGNORECASE)
    if not match:
        return line  # Nenhuma SQL encontrada, retorna a linha como está

    first_sql_index = match.start()

    # Divide a linha: antes e depois da SQL
    before_sql = line[:first_sql_index].strip()
    after_sql = line[first_sql_index:].strip()

    # Extrai variáveis ($var) da parte SQL
    variables = re.findall(r'(?<![\w])\$[a-zA-Z_]\w*', after_sql)
    
    # Monta nova linha
    new_line = before_sql + " SQL_QUERY"
    for var in variables:
        new_line += " " + var

    return new_line.strip()

def remove_strings(line):
    line = re.sub(r'`[^`]*`', 'exec', line)
    line = re.sub(r'(["\']).*?\1', '', line)
    return line

def process_arrays(line, arrays, file_id):
    array_pattern = re.compile(r'''
    =\s*array\s*\(       |   # Forma 1
    =\s*\[               |   # Forma 2
    \$\w+\s*\[\s*\]\s*=      # Forma 3
    ''', re.VERBOSE)

    access_pattern = re.compile(r'\$(\w+)\[(\d+)\]')
    match = array_pattern.search(line)
    if not match:
        # Tentar detetar o caso de acesso ao índice do array
        access_match = access_pattern.search(line)
        if access_match:
            var_name = access_match.group(1)
            index = access_match.group(2)
            # Transformar $array[1] em $array 1, mantendo a primeira parte da linha
            return line.replace(f"{var_name}[{index}]", f"{var_name} {index}")
        return line  # Se não for um array ou acesso, retorna a linha original

    # Extrair nome da variável
    var_match = re.match(r'\s*(\$\w+)', line)
    if not var_match:
        return line
    var = var_match.group(1)

    if var not in arrays:
        arrays[var] = {
            "var": var,
            "count": 0,
            "file_id": file_id
        }
        return line

    access_match = access_pattern.search(line)
    if access_match:
        var_name = access_match.group(1)
        index = access_match.group(2)
        # Transformar $array[1] em $array 1, mantendo a primeira parte da linha
        return line.replace(f"{var_name}[{index}]", f"{var_name} {index}")

    if var == arrays[var]["var"] and arrays[var]["file_id"] == file_id and line[len(var):len(var)+2] == "[]":
        second_half = line.split("=")[1].strip()
        new_line = var + " " + str(arrays[var]["count"]) + " " + "=" + " " + second_half
        arrays[var]["count"] += 1

    return new_line

def process_normal_line(line, indentation_count): #, dict
    
    line = obtain_query(line)
    line = remove_strings(line)
    
    pattern = r'\$?\w+(?:->\w+)?' #pattern to keep the $ symbol
    tokenized_line = regexp_tokenize(line.strip(), pattern)
    if tokenized_line == []:
        return ""
    
    # var = $var  == 'safe1' ? 'safe1' : var; ...
    p = r'\$\w+\s*=\s*\$[^;]+?\s*\?\s*[^;]+?\s*:\s*[^;]+?;'
    np = line.replace("(", "").replace(")", "")
    if re.search(p, np):
        if not tokenized_line[-1].startswith("$") and not tokenized_line[-2].startswith("$"):
            return " " * (indentation_count * 2) + tokenized_line[0]
        else: 
            if tokenized_line[-1].startswith("$"):
                return " " * (indentation_count * 2) + tokenized_line[0] + " " + tokenized_line[-1]
            else:
                return " " * (indentation_count * 2) + tokenized_line[0] + " " + tokenized_line[-2]
    
    
    # == ? :
    new_line = []
    for word in tokenized_line:
        if word != None and not "SQL_QUERY" in new_line:
           new_line.append(word)
        elif '$' in word or word.isdigit():
            new_line.append(word)


    # var = string
    p = r'\$([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:".+?"|sprintf\(".+?",\s*\$[a-zA-Z_][a-zA-Z0-9_]*\));'
    if re.search(p, line):
        new_line = obtain_query(tokenized_line)
        if "SQL_QUERY" in new_line:
            return " " * (indentation_count * 2) + new_line
        else:
            return " " * (indentation_count * 2) + tokenized_line[0]

    new_line = tokenized_line
        
    return " " * (indentation_count * 2) + ' '.join(new_line)

def process_echo(line, indentation_count):
    line = line.replace(".", " ")
    line = re.sub(r'(["\'`]).*?\1', '', line)
    line = line.strip()
    line = line.split()
    if line[-1] == ";":
        line = line[:-1]
    return " " * (indentation_count * 2) + ' '.join(line)

def is_query(line):
    sql_tokens = ["SELECT", "UPDATE", "DELETE", "INSERT", "CREATE", "ALTER", "DROP"]
    for token in sql_tokens:
        if token in line:
            return True
    return False

def is_var(line):
    return "$" in line.strip()

def php_line_processing(lines, php_status=None, file_id=None):
    codelines = []
    if php_status != None:
        if php_status == "Unsafe":
            codelines.append("1")
        else:
            codelines.append("0")

    arrays = {}
    indentation_count = 0
    oneline_if = False

    for line in lines:
        line = process_arrays(line, arrays, file_id)
        stripped_line = line.strip()

        if stripped_line.startswith(("if", "else", "elseif", "else if")):
            codelines.append(process_conditional_statement(line, indentation_count))
            if "{" in stripped_line:
                indentation_count += 1
            else:
                oneline_if = True
                indentation_count += 1

        elif stripped_line.startswith(("class", "public", "private", "function")):
            codelines.append(process_normal_line(line, indentation_count))
            if "{" in stripped_line:
                indentation_count += 1

        elif stripped_line.startswith(("for", "while")):
            loop_cond = process_normal_line(line, indentation_count)
            loop_cond_no_word = re.sub(r'^.*?\s+', '', loop_cond)
            codelines.append(" " * (indentation_count * 2) + "loop " + loop_cond_no_word)
            if "{" in stripped_line:
                indentation_count += 1

        elif stripped_line.startswith("do"):
            loop_cond = process_normal_line(line, indentation_count)
            loop_cond_no_word = re.sub(r'^.*?\s+', '', loop_cond)
            codelines.append(" " * (indentation_count * 2) + "do-loop " + loop_cond_no_word)
            if "{" in stripped_line:
                indentation_count += 1

        elif stripped_line.startswith("}"):
            indentation_count = max(0, indentation_count - 1)

            # Detecta se após o } vem else ou while (para do-while)
            if re.search(r'\}\s*(else|while)', stripped_line):
                keyword = re.search(r'\}\s*(else|while)', stripped_line).group(1)
                codelines.append("  " * indentation_count + keyword)
                if "{" in stripped_line:
                    indentation_count += 1
                else:
                    oneline_if = True
            else:
                codelines.append("")


        elif "endif" in stripped_line:
            indentation_count = max(0, indentation_count - 1)
            codelines.append("")

        elif stripped_line.startswith("echo"):
            codelines.append(process_echo(line, indentation_count))

        else:
            if stripped_line == "{":
                indentation_count += 1
            elif stripped_line == "}":
                indentation_count = max(0, indentation_count - 1)
            else:
                codelines.append(process_normal_line(line, indentation_count))
                if oneline_if:
                    indentation_count = max(0, indentation_count - 1)
                    oneline_if = False
    
    for index in range(0, len(codelines)):
        codelines[index] = remove_superglobal_prefix(codelines[index])

    #print("Processed lines:", codelines)
    #Delete empty lines
    codelines = [line for line in codelines if line.strip() != ""]
    obfuscated_codelines = preserve_indentation(codelines, obfuscate_variables)
    obfuscated_codelines = preserve_indentation(obfuscated_codelines, join_obs_index)
    obfuscated_codelines = preserve_indentation(obfuscated_codelines, obfuscate_classes_and_functions)
    return obfuscated_codelines