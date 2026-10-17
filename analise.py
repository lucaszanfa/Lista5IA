from pathlib import Path
from itertools import combinations
import json
import math
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'

def preparar():
    bruto = pd.read_csv(ROOT / 'titanic.csv')
    # Não usar o desfecho na imputação: mediana por sexo e classe.
    idade = bruto.age.fillna(bruto.groupby(['sex', 'pclass']).age.transform('median'))
    idade = idade.fillna(bruto.age.median())
    dados = pd.DataFrame({
        'Classe': bruto.pclass.map({1: '1a', 2: '2a', 3: '3a'}),
        'Sexo': bruto.sex.map({'female': 'Feminino', 'male': 'Masculino'}),
        'Idade': pd.cut(idade, [-np.inf, 18, 23, 41, np.inf], right=False,
                        labels=['0-17', '18-22', '23-40', '41+']),
        'Tarifa': pd.cut(bruto.fare.fillna(bruto.fare.median()),
                         [-np.inf, 10, 30, 100, np.inf], right=False,
                         labels=['<10', '10-<30', '30-<100', '100+']),
        'Embarque': bruto.embarked.fillna(bruto.embarked.mode()[0]),
        'Familia': pd.cut(bruto.sibsp + bruto.parch, [-1, 0, 3, np.inf],
                          labels=['Sozinho', '1-3', '4+']),
    })
    positivo = pd.get_dummies(dados, prefix_sep='=', dtype=bool)
    negativo = (~positivo).rename(columns=lambda s: 'NÃO ' + s)
    alvo = pd.DataFrame({'Sobreviveu=Sim': bruto.survived.eq(1),
                         'Sobreviveu=Não': bruto.survived.eq(0)})
    # Não duplicar o alvo por negação: NÃO Sim já corresponde a Não.
    binario = pd.concat([positivo, negativo, alvo], axis=1)
    return bruto, dados, binario

def apriori(binario, suporte=0.05, max_len=3):
    """Apriori: junção de prefixos e poda de subconjuntos infrequentes."""
    matriz = binario.to_numpy(dtype=bool)
    n = len(matriz)
    minimo = math.ceil(suporte * n - 1e-10)
    contagens = {}
    nivel = set()
    for j in range(matriz.shape[1]):
        qtd = int(matriz[:, j].sum())
        if qtd >= minimo:
            nivel.add((j,))
            contagens[(j,)] = qtd
    for k in range(2, max_len + 1):
        candidatos = set()
        ordenado = sorted(nivel)
        for i, a in enumerate(ordenado):
            for b in ordenado[i + 1:]:
                if a[:-1] != b[:-1]:
                    continue
                candidato = a + (b[-1],)
                if all(s in nivel for s in combinations(candidato, k - 1)):
                    candidatos.add(candidato)
        proximo = set()
        for c in candidatos:
            qtd = int(matriz[:, c].all(axis=1).sum())
            if qtd >= minimo:
                proximo.add(c)
                contagens[c] = qtd
        nivel = proximo
        if not nivel:
            break
    return contagens

def regras(binario, contagens, suporte, confianca):
    nomes = list(binario.columns)
    n = len(binario)
    todas = []
    for conjunto, qtd in contagens.items():
        if len(conjunto) < 2 or qtd / n + 1e-12 < suporte:
            continue
        for k in range(1, len(conjunto)):
            for a in combinations(conjunto, k):
                b = tuple(x for x in conjunto if x not in a)
                conf = qtd / contagens[a]
                if conf + 1e-12 >= confianca:
                    todas.append(dict(a=[nomes[x] for x in a], b=[nomes[x] for x in b],
                                      n=qtd, na=contagens[a], suporte=qtd / n,
                                      confianca=conf, lift=conf / (contagens[b] / n)))
    # Regra de interesse: apenas sobrevivência no consequente, sem alvo no antecedente.
    uteis = []
    for r in todas:
        if len(r['b']) != 1 or not r['b'][0].startswith('Sobreviveu='):
            continue
        if any(x.startswith('Sobreviveu=') for x in r['a']):
            continue
        atributos = [x.removeprefix('NÃO ').split('=')[0] for x in r['a']]
        if len(set(atributos)) != len(atributos) or r['lift'] <= 1 + 1e-12:
            continue
        uteis.append(r)
    # Antecedentes equivalentes na amostra são apresentados uma única vez.
    uteis.sort(key=lambda r: (sum(x.startswith('NÃO ') for x in r['a']), len(r['a']), r['a']))
    vistos = set()
    unicas = []
    for r in uteis:
        chave = (np.packbits(binario[r['a']].all(axis=1).to_numpy()).tobytes(), r['b'][0])
        if chave not in vistos:
            vistos.add(chave)
            unicas.append(r)
    unicas.sort(key=lambda r: (-r['lift'], -r['confianca'], -r['suporte'], r['a']))
    return todas, unicas

def executar():
    bruto, dados, binario = preparar()
    frequentes = apriori(binario)
    configuracoes = []
    escolhidas = None
    for sup, conf in [(0.05, 0.70), (0.10, 0.70), (0.10, 0.80), (0.15, 0.80), (0.10, 0.90)]:
        todas, uteis = regras(binario, frequentes, sup, conf)
        configuracoes.append(dict(suporte=sup, confianca=conf,
                                  itemsets=sum(v / len(bruto) + 1e-12 >= sup for v in frequentes.values()),
                                  regras=len(todas), uteis=len(uteis)))
        if (sup, conf) == (0.10, 0.80):
            escolhidas = uteis
            pd.DataFrame(todas).to_json(ROOT / 'regras_completas.json', orient='records', force_ascii=False, indent=2)
    resultado = dict(n=len(bruto), sobreviventes=int(bruto.survived.sum()),
                     ausentes=bruto.isna().sum().to_dict(), colunas=binario.shape[1],
                     sha256=hashlib.sha256((ROOT / 'titanic.csv').read_bytes()).hexdigest(),
                     configuracoes=configuracoes, regras=escolhidas)
    (ROOT / 'resultados.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    dados.to_csv(ROOT / 'dados_tratados.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(escolhidas).to_csv(ROOT / 'regras_selecionadas.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado

if __name__ == '__main__':
    executar()
