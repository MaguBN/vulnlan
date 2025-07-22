import os

# Pastas de origem e destino
pasta_origem = "./proc_func"         # <-- substitui pelo nome da pasta original
pasta_destino = "./proc_func_ss"    # <-- nova pasta onde vamos guardar os ficheiros truncados

# Criar pasta destino se não existir
os.makedirs(pasta_destino, exist_ok=True)

# Nome do sensitive sink
sensitive_sink = "mysql_query"

for ficheiro in os.listdir(pasta_origem):
    if ficheiro.endswith(".il"):
        caminho_origem = os.path.join(pasta_origem, ficheiro)
        caminho_destino = os.path.join(pasta_destino, ficheiro)

        with open(caminho_origem, "r", encoding="utf-8") as f:
            linhas = f.readlines()  # mantém a indentação e quebras de linha

        linhas_truncadas = []
        for linha in linhas:
            linhas_truncadas.append(linha)
            if sensitive_sink in linha:
                break  # parar logo após a primeira ocorrência do sensitive sink

        with open(caminho_destino, "w", encoding="utf-8") as f:
            f.writelines(linhas_truncadas)