from pathlib import Path
import base64
import contextlib
import gzip
import io
import json
import random
import textwrap
import nbformat
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import A4
import pymupdf

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / 'resultados.json').read_text(encoding='utf-8'))

def notebook():
    fonte = (ROOT / 'analise.py').read_text(encoding='utf-8')
    fonte = fonte[:fonte.index("if __name__ == '__main__':")]
    fonte = fonte.replace('Path(__file__).resolve().parent', 'Path.cwd()')
    inicio, restante = fonte.split('def preparar():', 1)
    preparo, restante = ('def preparar():' + restante).split('def apriori(', 1)
    apriori, restante = ('def apriori(' + restante).split('def regras(', 1)
    regras, executar = ('def regras(' + restante).split('def executar():', 1)
    snapshot = base64.b64encode(gzip.compress((ROOT / 'titanic.csv').read_bytes())).decode()
    cells = []
    def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s): cells.append(nbformat.v4.new_code_cell(s))
    md('# Lista 5 — Questão 07: Apriori e Titanic\n\nResolução comentada e reproduzível. '
       'Execute as células em ordem. Dependências: Python 3, pandas e numpy. '
       'O algoritmo Apriori está implementado abaixo, sem depender de mlxtend.\n\n'
       '**Nota sobre o material:** o Apriori.ipynb da disciplina não estava entre os arquivos '
       'fornecidos. Esta é uma implementação independente do mesmo algoritmo.\n\n'
       '**Base:** [seaborn-data/titanic.csv](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv), '
       '891 passageiros. Uma cópia compactada está incorporada para execução sem internet.')
    code(inicio)
    md('## 1. Cópia reprodutível da base\nSe o CSV não estiver ao lado do notebook, '
       'restauramos a cópia utilizada na resolução. O hash evita trocar silenciosamente a base.')
    code("import base64, gzip\n"
         "snapshot = " + repr(snapshot) + "\n"
         "if not (ROOT / 'titanic.csv').exists():\n"
         "    (ROOT / 'titanic.csv').write_bytes(gzip.decompress(base64.b64decode(snapshot)))\n"
         "assert hashlib.sha256((ROOT / 'titanic.csv').read_bytes()).hexdigest() == " + repr(R['sha256']) + "\n"
         "print('Base conferida: 891 passageiros.')")
    md('## 2. Tratamento e discretização\n177 idades ausentes: mediana por sexo e classe, '
       'com mediana global de reserva; o alvo não participa da imputação. Dois portos ausentes: moda S. '
       'Idade: [0,18), [18,23), [23,41), [41,+∞); tarifa: [0,10), [10,30), [30,100), [100,+∞). '
       'Familia = sibsp + parch, sem o próprio passageiro, em 0, 1–3 e 4+. '
       'As fronteiras foram escolhidas antes da mineração. Removemos campos redundantes, derivados '
       'e deck (688 ausentes). alive replica o alvo e causaria vazamento.')
    code(preparo + "\nbruto, dados, binario = preparar()\nprint('Dimensões:', binario.shape)\n"
         "print('Ausentes na matriz:', int(binario.isna().sum().sum()))\n"
         "print(dados.head().to_string(index=False))\n"
         "print('\\nExemplos de itens:', ', '.join(binario.columns[:8]))")
    md('## 3. Presença e ausência\nPara cada indicador x, NÃO x = 1 − x. '
       'São 19 indicadores de perfil, 19 complementos e dois indicadores do alvo: 40 colunas. '
       'As negações são feitas após tratar os ausentes. NÃO Classe=1a significa classe 2a ou 3a; '
       'não significa dado faltante. Não criamos negações extras do alvo.')
    code("perfil = [x for x in binario if not x.startswith(('NÃO ', 'Sobreviveu='))]\n"
         "assert all((binario[x] != binario['NÃO ' + x]).all() for x in perfil)\n"
         "print('Complementos verificados:', len(perfil))\n"
         "print(bruto.groupby(['sex', 'pclass']).age.median().to_string())")
    md('## 4. Apriori\nJunção de itemsets frequentes e poda: um candidato só é contado '
       'se todos os subconjuntos de tamanho k−1 forem frequentes. Limitamos max_len=3 '
       'para apresentar antecedentes com até dois itens e um alvo. A busca não cobre regras mais longas.')
    code(apriori)
    md('## 5. Métricas e filtros\nSuporte = n(A e B)/N; confiança = n(A e B)/n(A); '
       'lift = confiança/P(B). Geramos primeiro todas as partições dos itemsets frequentes. '
       'Depois mantemos apenas consequente unitário de sobrevivência, sem o alvo no antecedente, '
       'atributos distintos no antecedente e lift > 1. Antecedentes com a mesma máscara de passageiros '
       'são apresentados uma vez, dando preferência a menos negações. É uma redução empírica; '
       'não equivale a uma prova de equivalência lógica ou significância estatística.')
    code(regras + '\ndef executar():' + executar)
    code("resultado = executar()\nprint('\nComparação:')\nprint(pd.DataFrame(resultado['configuracoes']).to_string(index=False))".replace("print('\nComparação", "print('\\nComparação"))
    md('## 6. Regras ordenadas por lift\nConfiguração escolhida: suporte 0,10 e confiança 0,80. '
       'A tabela apresenta todas as 36 regras após os filtros. Percentuais são calculados sobre a base tratada.')
    code("tabela = pd.DataFrame(resultado['regras'])\n"
         "tabela['antecedente'] = tabela.a.map(lambda a: ' E '.join(a))\n"
         "tabela['consequente'] = tabela.b.map(lambda b: b[0])\n"
         "print(tabela[['antecedente','consequente','n','na','suporte','confianca','lift']].to_string(index=False))")
    md('## 7. Interpretação\n'
       '1. Mulheres da 1a classe: 91/94 sobreviveram; confiança 96,81%, lift 2,522.\n'
       '2. Mulheres NÃO da 3a classe: 161/170; confiança 94,71%, lift 2,467.\n'
       '3. Mulheres que NÃO embarcaram em S: 91/109; confiança 83,49%, lift 2,175.\n'
       '4. Homens com tarifa <10: 243/272 não sobreviveram; confiança 89,34%, lift 1,450.\n'
       '5. Homens da 3a classe: 300/347 não sobreviveram; confiança 86,46%, lift 1,403.\n'
       '6. Tarifa <10 e NÃO idade 18–22: 205/240 não sobreviveram; confiança 85,42%, lift 1,386.\n\n'
       'As negações agrupam categorias e não devem ser interpretadas como causas. Classe, sexo, '
       'porto e tarifa podem estar associados entre si. As idades imputadas podem mudar de faixa '
       'e afetar as regras; os resultados são descritivos desta amostra, sem validação externa.')
    md('## 8. Conferência independente das métricas\nRecontamos diretamente no CSV binarizado '
       'cada uma das regras publicadas, sem consultar as contagens do Apriori.')
    code("for r in resultado['regras']:\n"
         "    a = binario[r['a']].all(axis=1)\n"
         "    b = binario[r['b']].all(axis=1)\n"
         "    n_ab = int((a & b).sum())\n"
         "    assert n_ab == r['n'] and int(a.sum()) == r['na']\n"
         "    assert abs((a & b).mean() - r['suporte']) < 1e-12\n"
         "    assert abs(n_ab / a.sum() - r['confianca']) < 1e-12\n"
         "    assert abs((n_ab / a.sum()) / b.mean() - r['lift']) < 1e-12\n"
         "assert len(resultado['regras']) == 36\n"
         "print('36 regras conferidas por contagem direta. Nenhum valor ausente na matriz.')")
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    namespace = {'__name__': '__notebook__'}
    count = 0
    for cell in nb.cells:
        if cell.cell_type != 'code': continue
        count += 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(cell.source, '<notebook>', 'exec'), namespace)
        cell.execution_count = count
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=output.getvalue())] if output.getvalue() else []
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / 'Lista5_resolvida.ipynb')

pdfmetrics.registerFont(TTFont('Mao', 'C:/Windows/Fonts/segoepr.ttf'))
pdfmetrics.registerFont(TTFont('MaoB', 'C:/Windows/Fonts/segoeprb.ttf'))
W, H = A4
AZUL = HexColor('#23477e')
random.seed(51)
c = canvas.Canvas(str(ROOT / 'Lista5_resolvida.pdf'), pagesize=A4)
c.setTitle('Lista 5 resolvida - Apriori e Titanic')
c.setAuthor('Resolução de estudo')

def escrita(texto, x, y, size=12, bold=False):
    font = 'MaoB' if bold else 'Mao'
    for letra in texto:
        c.saveState()
        c.translate(x, y + random.uniform(-.38, .38))
        c.rotate(random.uniform(-.7, .7))
        c.setFillColor(AZUL)
        c.setFont(font, size + random.uniform(-.12, .12))
        c.drawString(0, 0, letra)
        c.restoreState()
        x += pdfmetrics.stringWidth(letra, font, size) + random.uniform(-.07, .07)

PASSO = 23

def linha(texto='', size=12, bold=False, indent=0):
    global y
    if y < 55: raise ValueError('Conteúdo excedeu a página: ' + texto)
    escrita(texto, 68 + indent, y, size, bold)
    y -= PASSO

def par(texto, size=12):
    # Quebra por largura real da fonte manuscrita.
    words = texto.split()
    out = ''
    for word in words:
        nxt = (out + ' ' + word).strip()
        if pdfmetrics.stringWidth(nxt, 'Mao', size) > W - 105:
            linha(out, size)
            out = word
        else: out = nxt
    if out: linha(out, size)

def space():
    global y
    y -= 5

def page(n, titulo):
    global y
    if n > 1: c.showPage()
    c.setFillColor(HexColor('#fffef9'))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setLineWidth(.4)
    c.setStrokeColor(HexColor('#dce7ef'))
    for yy in range(62, 790, 23): c.line(32, yy, W-27, yy)
    c.setStrokeColor(HexColor('#e8b5b2'))
    c.line(54, 43, 54, H-35)
    escrita('Lista 5 • Inteligência Artificial', 68, H-43, 11)
    escrita(str(n) + '/8', W-60, 28, 10)
    y = H-79
    linha(titulo, 17, True)
    space()

def pct(x): return f'{100*x:.2f}%'.replace('.', ',')
def num(x): return f'{x:.3f}'.replace('.', ',')
def nome(s):
    return s.replace('Sexo=Feminino','Mulher').replace('Sexo=Masculino','Homem').replace('Classe=', 'Classe ').replace('Idade=', 'Idade ').replace('Tarifa=', 'Tarifa ').replace('Embarque=', 'Porto ').replace('Familia=', 'Família ')
def regra(r): return ' + '.join(nome(x) for x in r['a'])

page(1, 'Questão 07 — Titanic e Apriori')
par('Objetivo: encontrar perfis de passageiros associados à sobrevivência, usando itens presentes e ausentes.')
space()
linha('1) Preparação da base', 14, True)
par('Usei titanic.csv do repositório seaborn-data: 891 passageiros, 342 sobreviventes e 549 não sobreviventes. Nenhuma linha foi excluída.')
par('Idade: preenchi 177 ausências com a mediana do grupo sexo × classe. A mediana global fica como reserva. A sobrevivência não entra nessa imputação.')
par('Embarque: preenchi os 2 ausentes com S, a moda. A tarifa não tinha ausências; o código prevê a mediana se necessário.')
space()
linha('Atributos escolhidos', 13, True)
par('Classe, sexo, idade, tarifa, porto de embarque e número de familiares a bordo. O alvo é Sobreviveu=Sim/Não.')
par('Família = sibsp + parch: irmãos/cônjuge + pais/filhos; o próprio passageiro não entra nessa soma.')
par('Excluí deck, com 688 ausências. Removi class, embark_town, alone, who e adult_male por redundância ou derivação. alive replica o alvo e causaria vazamento.')
par('Esta versão já não tem nomes nem identificadores. Se existissem, também seriam removidos.')
space()
par('Material: o Apriori.ipynb da disciplina não foi fornecido; usei uma implementação própria e comentada do Apriori.', 11)

page(2, 'Discretização e itens de ausência')
linha('1b) Faixas usadas antes de minerar', 14, True)
par('Idade: [0,18), [18,23), [23,41) e [41,+infinito). São as faixas 0–17, 18–22, 23–40 e 41+; idades fracionárias entram sem arredondamento.')
par('A separação distingue menores de idade, jovens, adultos e adultos mais velhos, seguindo a sugestão do enunciado.')
par('Tarifa: [0,10), [10,30), [30,100) e [100,+infinito), em libras. São faixas amplas de preço; não são classes sociais nem quantis.')
par('Família: 0 (sozinho), 1–3 e 4+. Diferenciam ausência de parentes, grupos pequenos e grupos maiores.')
par('As fronteiras são escolhas interpretáveis, não valores otimizados pelo desfecho. Outras faixas podem mudar as regras.')
space()
linha('1c e 2) Codificação binária', 14, True)
par('Cada atributo=valor vira uma coluna booleana: Classe=1a, Idade=18-22 etc. Para cada coluna x de perfil, criei o complemento NÃO x = 1 − x.')
par('Exemplo: um passageiro da 2a classe tem Classe=2a igual a 1 e NÃO Classe=1a igual a 1.')
par('NÃO Idade=18-22 reúne as outras três faixas. Ausência da característica não significa valor desconhecido: os dados já foram tratados.')
par('Total: 19 itens de perfil + 19 negações + 2 itens de sobrevivência = 40 colunas, em 891 transações.')
par('Não dupliquei negações do alvo: NÃO Sobreviveu=Sim seria o mesmo que Sobreviveu=Não.')

page(3, 'Execução e escolha dos parâmetros')
par('3a) O Apriori conta itens, combina os frequentes e poda candidatos com subconjuntos infrequentes. Usei itemsets de tamanho até 3 para obter antecedentes com até 2 itens.')
par('Para cada itemset frequente, gerei todas as partições A → B, respeitando suporte e confiança mínimos.')
space()
linha('3b) Comparação das configurações', 14, True)
linha('Sup.   Conf.    Itemsets    Todas    Filtradas', 11, True)
for cfg in R['configuracoes']:
    linha(f"{int(cfg['suporte']*100):2d}%     {int(cfg['confianca']*100)}%      {cfg['itemsets']}       {cfg['regras']}       {cfg['uteis']}",11)
space()
par('“Todas” inclui regras entre quaisquer itens. “Filtradas” mantém só sobrevivência no consequente, sem alvo no antecedente, lift > 1 e atributos diferentes no antecedente.')
par('Também reuni antecedentes que selecionam os mesmos passageiros, preferindo menos negações. Isso evita repetir Homem e NÃO Mulher, por exemplo. É equivalência na amostra.')
par('Escolhi suporte 10% e confiança 80%: pelo menos 90 passageiros satisfazem a regra completa, com 36 regras interpretáveis. A 90% de confiança restam só 2; suporte 15% perde perfis menores.')
space()
linha('3c) Métricas', 14, True)
linha('sup(A → B) = n(A e B) / 891', 12)
linha('conf(A → B) = n(A e B) / n(A)', 12)
linha('lift(A → B) = conf(A → B) / P(B)', 12)
par('Bases: P(Sim)=38,38%; P(Não)=61,62%. Lift > 1 indica associação positiva com o consequente.',11)

page(4, '4) Regras de sobrevivência')
par('As cinco regras com Sobreviveu=Sim, em ordem decrescente de lift. “n” conta quem satisfaz antecedente e consequente; “A” conta só o antecedente.',11)
sim = [r for r in R['regras'] if r['b']==['Sobreviveu=Sim']]
for i,r in enumerate(sim,1):
    space()
    linha(f'S{i}. {regra(r)} → Sim', 11.7, True)
    linha(f"n/A = {r['n']}/{r['na']} | sup {pct(r['suporte'])} | conf {pct(r['confianca'])}",11)
    linha(f"lift = {num(r['lift'])}",11)
space()
linha('Interpretações',14,True)
par('S1: 91 das 94 mulheres da 1a classe sobreviveram. A proporção é 2,52 vezes a taxa de sobrevivência da base inteira.')
par('S2 (ausência): mulheres NÃO da 3a classe são as da 1a ou 2a. Sobreviveram 161 de 170; o grupo reúne duas classes, sem atribuir uma causa à classe.')
par('S3 (ausência): entre mulheres que NÃO embarcaram em Southampton, 91 de 109 sobreviveram. O grupo reúne C e Q; porto pode refletir outros fatores, como classe e tarifa.')
par('S4 e S5: mulheres de 23–40 anos e mulheres com 1–3 familiares também apresentam sobrevivência acima de 80%. As regras se sobrepõem; seus suportes não devem ser somados.',11.5)

page(5, '4) Regras de não sobrevivência')
nao = [r for r in R['regras'] if r['b']==['Sobreviveu=Não']]
par('As cinco primeiras regras abaixo são as de maior lift para Sobreviveu=Não.',11)
for i,r in enumerate(nao[:5],1):
    space()
    linha(f'N{i}. {regra(r)} → Não',11.7,True)
    linha(f"n/A = {r['n']}/{r['na']} | sup {pct(r['suporte'])} | conf {pct(r['confianca'])}",11)
    linha(f"lift = {num(r['lift'])}",11)
space()
linha('Interpretações',14,True)
par('N1: 243 de 272 homens com tarifa inferior a 10 não sobreviveram (89,34%). O lift 1,450 compara essa taxa aos 61,62% de não sobrevivência da base.')
par('N3: na 3a classe, 300 de 347 homens não sobreviveram. A confiança de 86,46% mostra um perfil de alta mortalidade nesta amostra.')
par('N5 (ausência): tarifa <10 e NÃO idade 18–22 agrupa menores de 18 e passageiros de 23 anos ou mais. Houve 205 mortes entre 240 passageiros desse perfil.')
par('As regras de morte têm lift menor que as de sobrevivência porque a taxa de morte de referência já é maior. Confiança alta, sozinha, não basta para avaliar uma regra.',11.5)

for pagina, ini, fim in [(6,0,18),(7,18,36)]:
    PASSO = 17
    page(pagina, f'Todas as regras selecionadas ({ini+1}–{fim})')
    par('Ordem global por lift. S = Sobreviveu Sim; N = Sobreviveu Não. Valores: suporte (%) / confiança (%) / lift.',10.8)
    space()
    for idx in range(ini,fim):
        r=R['regras'][idx]
        alvo='S' if r['b']==['Sobreviveu=Sim'] else 'N'
        linha(f'{idx+1:02d}. {regra(r)} → {alvo}',10.4)
        linha(f"     {pct(r['suporte'])}   /   {pct(r['confianca'])}   /   {num(r['lift'])}",10.5)

PASSO = 23
page(8, 'Conclusões e reprodução')
par('Nesta amostra, sexo e classe aparecem em perfis associados à sobrevivência. Mulheres da 1a ou 2a classe têm alta sobrevivência; homens com tarifas baixas e da 3a classe têm alta não sobrevivência.')
par('Os itens de ausência ampliam a leitura dos grupos: NÃO 3a classe reúne 1a e 2a, enquanto NÃO 18–22 reúne três faixas etárias. O significado de cada complemento deve ficar explícito.')
space()
linha('Limitações da análise',14,True)
par('Associação não demonstra causalidade. Não houve teste de significância, correção por múltiplas comparações ou validação em outra amostra. Regras podem compartilhar os mesmos passageiros.')
par('A imputação pode deslocar idades entre faixas. As escolhas de discretização, suporte e limite de tamanho afetam o resultado; a mineração é descritiva, não um modelo validado de previsão.')
par('Os filtros reduzem regras triviais e duplicadas, mas não eliminam toda redundância. As 36 regras estão integralmente nas páginas 6–7 e no arquivo CSV.')
space()
linha('Notebook comentado e executado',14,True)
notebook_url = 'https://github.com/lucaszanfa/Lista5IA/blob/main/Lista5_resolvida.ipynb'
link_top = y + 16
linha('Abrir notebook no GitHub (clique aqui)',11.5,True)
# Endereço inteiro em uma única operação de texto: permite selecionar/copiar
# e reconhecer a URL em leitores que ignoram as anotações do PDF.
c.setFillColor(AZUL)
c.setFont('Mao', 9)
c.drawString(68, y, notebook_url)
url_width = pdfmetrics.stringWidth(notebook_url, 'Mao', 9)
c.setStrokeColor(AZUL)
c.setLineWidth(.6)
c.line(68, y - 3, 68 + url_width, y - 3)
c.linkURL(notebook_url, (65, y - 8, W - 35, link_top),
          relative=0, thickness=0)
y -= 23
par('O notebook está disponível publicamente no endereço acima, com código comentado, saídas executadas e uma cópia incorporada da base.',11)
space()
linha('Fonte dos dados (link)',13,True)
c.linkURL('https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv', (65,y+17,270,y+39), thickness=0)
linha('github.com/mwaskom/seaborn-data',11)
linha('Arquivo: titanic.csv • acesso em 17/10/2026',11)
par('Conferência: as 36 regras foram recontadas diretamente na matriz binária; suporte, confiança e lift coincidiram com as saídas do Apriori.',11)
c.save()

if __name__ == '__main__':
    notebook()
    doc = pymupdf.open(ROOT / 'Lista5_resolvida.pdf')
    assert len(doc)==8
    doc[3].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(ROOT / 'previa_pagina4.png')
    doc[6].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(ROOT / 'previa_pagina7.png')
    print('PDF de 8 páginas e notebook executado gerados.')
