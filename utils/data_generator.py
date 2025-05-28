import random
import pandas as pd

generos = ["masculino", "feminino"]
fumante_opcoes = ["sim", "não"]
regioes = ["nordeste", "noroeste", "sudeste", "sudoeste", "sul"]


def gerar_dados_sinteticos(n, caminho_saida="dados_sinteticos.csv"):
    dados = []
    for _ in range(n):
        idade = random.randint(18, 65)
        genero = random.choice(generos)
        imc = round(random.uniform(18.5, 35.0), 2)
        filhos = random.randint(0, 5)
        fumante = random.choice(fumante_opcoes)
        regiao = random.choice(regioes)

        ruido = random.gauss(0, 2000)  # distribuição normal
        encargos = round(
            2000 + idade * 100 + imc * 50 + filhos * 300 +
            (10000 if fumante == "sim" else 0) + ruido,
            2
        )

        dados.append({
            "idade": idade,
            "gênero": genero,
            "imc": imc,
            "filhos": filhos,
            "fumante": fumante,
            "região": regiao,
            "encargos": encargos
        })

    df = pd.DataFrame(dados)
    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"Arquivo '{caminho_saida}' gerado com sucesso com {n} registros.")


gerar_dados_sinteticos(200, caminho_saida="dados_teste.csv")
