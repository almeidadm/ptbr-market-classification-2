"""
Detectores de duplicatas exatas e near-duplicates para a EDA §7.

Duplicatas exatas: contagem de repetições de `link`, `title`, `text` com
amostras representativas por tipo.

Near-duplicates: MinHash + LSH sobre shingles de caracteres (`k=5`) do
campo `text` normalizado (lower + colapso de whitespace). Threshold
Jaccard padrão 0,85, conforme decisão 002.

A política de descarte de `text` nulo é aplicada **pelo chamador** — este
módulo opera sobre o DataFrame recebido e descarta apenas internamente
linhas com texto vazio após normalização (não há o que "shingle-ar").
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

import pandas as pd

SHINGLE_K = 5
THRESHOLD_PADRAO = 0.85
NUM_PERM_PADRAO = 128

_REGEX_WHITESPACE = re.compile(r"\s+")


def _normalizar_texto(texto: str) -> str:
    return _REGEX_WHITESPACE.sub(" ", texto.lower()).strip()


def _shingles(texto: str, k: int = SHINGLE_K) -> set[str]:
    if len(texto) < k:
        return {texto} if texto else set()
    return {texto[i : i + k] for i in range(len(texto) - k + 1)}


def duplicatas_exatas(
    df: pd.DataFrame,
    coluna: str,
    n_exemplos: int = 5,
) -> dict:
    """Detecta duplicatas exatas numa coluna e retorna resumo + amostras.

    `linhas_duplicadas` conta as ocorrências além da primeira em cada grupo
    (i.e., quantas linhas seriam perdidas se "manter a primeira" fosse a
    política). `grupos` conta quantos valores distintos aparecem em mais
    de uma linha. `exemplos` contém até `n_exemplos` grupos com o valor
    repetido e as `id`s (posições originais) das linhas envolvidas.
    """
    if coluna not in df.columns:
        raise KeyError(f"coluna não encontrada: {coluna}")

    nao_nulos = df[coluna].dropna()
    contagens = nao_nulos.value_counts()
    grupos_mask = contagens > 1
    contagens_dup = contagens[grupos_mask]

    if contagens_dup.empty:
        return {
            "tipo": coluna,
            "linhas_totais_nao_nulas": int(nao_nulos.size),
            "linhas_duplicadas": 0,
            "grupos": 0,
            "exemplos": [],
        }

    exemplos = []
    for valor in contagens_dup.head(n_exemplos).index:
        ids = df.index[df[coluna] == valor].tolist()
        texto_exibicao = str(valor)
        if len(texto_exibicao) > 160:
            texto_exibicao = texto_exibicao[:157] + "..."
        exemplos.append(
            {
                "valor": texto_exibicao,
                "n_ocorrencias": int(contagens_dup[valor]),
                "ids": [int(x) for x in ids[:10]],
            }
        )

    linhas_duplicadas = int((contagens_dup - 1).sum())
    return {
        "tipo": coluna,
        "linhas_totais_nao_nulas": int(nao_nulos.size),
        "linhas_duplicadas": linhas_duplicadas,
        "grupos": int(contagens_dup.size),
        "exemplos": exemplos,
    }


def resumo_duplicatas_exatas(df: pd.DataFrame, colunas: Iterable[str]) -> pd.DataFrame:
    """Wrapper tabular: aplica `duplicatas_exatas` a cada coluna e retorna DataFrame.

    O DataFrame tem uma linha por tipo (coluna) com: `tipo, linhas_duplicadas,
    grupos, pct_corpus, exemplos_json`. `pct_corpus` usa o total de linhas
    (com nulos) como denominador — é a "fração do corpus afetada" operacional.
    """
    total_corpus = len(df)
    linhas = []
    for coluna in colunas:
        relatorio = duplicatas_exatas(df, coluna)
        linhas.append(
            {
                "tipo": relatorio["tipo"],
                "linhas_duplicadas": relatorio["linhas_duplicadas"],
                "grupos": relatorio["grupos"],
                "pct_corpus": (
                    relatorio["linhas_duplicadas"] / total_corpus
                    if total_corpus > 0
                    else 0.0
                ),
                "exemplos_json": json.dumps(relatorio["exemplos"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(linhas)


def _construir_minhash_worker(args):
    """Worker de `multiprocessing.Pool`: constrói MinHash de um único texto.

    Precisa estar no nível do módulo para ser picklable. Retorna
    `(indice, MinHash | None)`; `None` sinaliza linha sem shingles válidos
    (texto nulo ou string vazia após normalização).
    """
    from datasketch import MinHash

    indice, texto, num_perm, k_shingle = args
    if not isinstance(texto, str):
        return indice, None
    normalizado = _normalizar_texto(texto)
    conjunto = _shingles(normalizado, k=k_shingle)
    if not conjunto:
        return indice, None
    mh = MinHash(num_perm=num_perm)
    for shingle in conjunto:
        mh.update(shingle.encode("utf-8"))
    return indice, mh


def _construir_minhashes(
    textos: pd.Series,
    num_perm: int,
    k_shingle: int,
    n_processos: int | None = None,
    progresso_a_cada: int = 20000,
):
    """Cria MinHash para cada linha (pulando linhas sem shingles).

    Retorna `dict[int, MinHash]` com a chave igual ao índice original (que
    é a posição pós-reset em `carregar_articles`, útil como ID).

    Quando `n_processos > 1`, paraleliza a construção com `multiprocessing.Pool`.
    O output é determinístico e byte-idêntico ao caminho serial: todos os
    workers criam `MinHash` com o seed default (permutações idênticas), e a
    ordenação final em `encontrar_near_duplicates` é por `(id_a, id_b)`.
    `n_processos=None` → `max(1, os.cpu_count() - 1)`.
    """
    if n_processos is None:
        n_processos = max(1, (os.cpu_count() or 1) - 1)

    if n_processos <= 1:
        from datasketch import MinHash

        minhashes: dict[int, "MinHash"] = {}
        contador = 0
        for indice, texto in textos.items():
            if not isinstance(texto, str):
                continue
            normalizado = _normalizar_texto(texto)
            conjunto = _shingles(normalizado, k=k_shingle)
            if not conjunto:
                continue
            mh = MinHash(num_perm=num_perm)
            for shingle in conjunto:
                mh.update(shingle.encode("utf-8"))
            minhashes[int(indice)] = mh
            contador += 1
            if contador % progresso_a_cada == 0:
                print(
                    f"[§7] MinHashes construídos (serial): {contador:,}".replace(",", "."),
                    flush=True,
                )
        return minhashes

    from multiprocessing import Pool

    tarefas = ((int(indice), texto, num_perm, k_shingle) for indice, texto in textos.items())
    # chunksize: cada worker recebe blocos grandes o suficiente para amortizar
    # o overhead de IPC, mas pequenos o suficiente para balancear carga entre
    # workers quando há variância no tamanho dos textos.
    chunksize = max(1, len(textos) // (n_processos * 8))
    print(
        f"[§7] paralelizando MinHashes: {n_processos} processos, chunksize={chunksize}",
        flush=True,
    )
    minhashes: dict[int, "MinHash"] = {}
    contador = 0
    with Pool(processes=n_processos) as pool:
        for indice, mh in pool.imap_unordered(
            _construir_minhash_worker, tarefas, chunksize=chunksize
        ):
            contador += 1
            if mh is not None:
                minhashes[indice] = mh
            if contador % progresso_a_cada == 0:
                print(
                    f"[§7] MinHashes processados: {contador:,} "
                    f"(válidos: {len(minhashes):,})".replace(",", "."),
                    flush=True,
                )
    return minhashes


def encontrar_near_duplicates(
    df: pd.DataFrame,
    coluna_texto: str = "text",
    threshold: float = THRESHOLD_PADRAO,
    num_perm: int = NUM_PERM_PADRAO,
    k_shingle: int = SHINGLE_K,
    progresso_a_cada: int = 20000,
    n_processos: int | None = None,
) -> pd.DataFrame:
    """Executa MinHash + LSH sobre `coluna_texto` e retorna pares candidatos.

    Retorna DataFrame com colunas `id_a, id_b, jaccard_estimado` onde `id_*`
    são índices do DataFrame de entrada (após reset) e `jaccard_estimado` é
    `mh_a.jaccard(mh_b)` — aproximação MinHash, não Jaccard exato.

    O LSH usa `threshold` para decidir colisões candidatas; filtramos à
    mão por `jaccard_estimado >= threshold` para remover falsos positivos
    da LSH (que prioriza recall).

    `n_processos` controla a paralelização da construção de MinHashes
    (única fase CPU-bound que paraleliza sem custo de correção). `None` →
    `max(1, os.cpu_count() - 1)`. Output byte-idêntico ao caminho serial.

    Emite prints de progresso a cada `progresso_a_cada` linhas processadas.
    """
    from datasketch import MinHashLSH

    textos = df[coluna_texto]
    print(f"[§7] construindo MinHashes para {len(textos):,} linhas...".replace(",", "."), flush=True)
    minhashes = _construir_minhashes(
        textos,
        num_perm=num_perm,
        k_shingle=k_shingle,
        n_processos=n_processos,
        progresso_a_cada=progresso_a_cada,
    )
    print(f"[§7] {len(minhashes):,} MinHashes construídos (linhas com texto válido).".replace(",", "."), flush=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    contador_insercoes = 0
    for indice, mh in minhashes.items():
        lsh.insert(f"row-{indice}", mh)
        contador_insercoes += 1
        if contador_insercoes % progresso_a_cada == 0:
            print(f"[§7] LSH insert: {contador_insercoes:,}".replace(",", "."), flush=True)

    print(f"[§7] consultando LSH para identificar pares...", flush=True)
    pares: dict[tuple[int, int], float] = {}
    contador_consultas = 0
    for indice, mh in minhashes.items():
        for chave_vizinho in lsh.query(mh):
            outro_indice = int(chave_vizinho.removeprefix("row-"))
            if outro_indice == indice:
                continue
            par = (
                (indice, outro_indice)
                if indice < outro_indice
                else (outro_indice, indice)
            )
            if par in pares:
                continue
            jaccard_estimado = mh.jaccard(minhashes[outro_indice])
            if jaccard_estimado >= threshold:
                pares[par] = float(jaccard_estimado)
        contador_consultas += 1
        if contador_consultas % progresso_a_cada == 0:
            print(
                f"[§7] LSH query: {contador_consultas:,} / {len(minhashes):,} "
                f"({len(pares):,} pares)".replace(",", "."),
                flush=True,
            )

    print(
        f"[§7] {len(pares):,} pares near-duplicate encontrados (threshold {threshold}).".replace(",", "."),
        flush=True,
    )

    if not pares:
        return pd.DataFrame(columns=["id_a", "id_b", "jaccard_estimado"])

    # Ordenação estável por `(id_a, id_b)` para reprodutibilidade do CSV.
    itens = sorted(pares.items(), key=lambda kv: kv[0])
    return pd.DataFrame(
        [
            {"id_a": int(a), "id_b": int(b), "jaccard_estimado": float(j)}
            for (a, b), j in itens
        ]
    )


def componentes_conexos(pares: pd.DataFrame) -> list[list[int]]:
    """Componentes conexos da união dos pares (union-find simples).

    Cada componente é uma lista ordenada de IDs. Componentes com tamanho 1
    não são retornados (pares são por definição conexos em pares).
    """
    pai: dict[int, int] = {}

    def achar(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(x: int, y: int) -> None:
        rx, ry = achar(x), achar(y)
        if rx != ry:
            pai[rx] = ry

    for _, linha in pares.iterrows():
        a, b = int(linha["id_a"]), int(linha["id_b"])
        if a not in pai:
            pai[a] = a
        if b not in pai:
            pai[b] = b
        unir(a, b)

    grupos: dict[int, list[int]] = {}
    for no in pai:
        raiz = achar(no)
        grupos.setdefault(raiz, []).append(no)
    return [sorted(g) for g in grupos.values() if len(g) > 1]


def notificar_afetadas_near(pares: pd.DataFrame) -> set[int]:
    """Conjunto de IDs que seriam descartados se mantivéssemos 1 linha por componente.

    Para cada componente conexo de tamanho `n`, `n-1` linhas são "afetadas"
    (duplicadas em relação à canônica). Retorna o conjunto dessas IDs.
    """
    afetadas: set[int] = set()
    for grupo in componentes_conexos(pares):
        # Mantém o menor ID como canônico; os demais são "afetados".
        for id_duplicado in grupo[1:]:
            afetadas.add(int(id_duplicado))
    return afetadas


def notificar_afetadas_exatas(df: pd.DataFrame, colunas: Iterable[str]) -> set[int]:
    """Conjunto de IDs que seriam descartados em duplicatas exatas.

    Para cada coluna, agrupa por valor e marca como "afetado" todo índice
    exceto o primeiro de cada grupo. A união das três colunas entra no
    conjunto — qualquer critério suficiente condena a linha.
    """
    afetadas: set[int] = set()
    for coluna in colunas:
        if coluna not in df.columns:
            continue
        sub = df[coluna].dropna()
        duplicadas_mask = sub.duplicated(keep="first")
        afetadas.update(int(x) for x in sub.index[duplicadas_mask].tolist())
    return afetadas


# API legada mantida para não quebrar chamadas antigas (usada em scripts
# que tenham sido construídos antes deste ciclo). Preferir `encontrar_near_duplicates`
# e `resumo_duplicatas_exatas` para novos usos.
def detectar_duplicatas(
    df: pd.DataFrame,
    threshold: float = THRESHOLD_PADRAO,
    coluna_texto: str = "text",
    num_perm: int = NUM_PERM_PADRAO,
    k_shingle: int = SHINGLE_K,
) -> dict:
    """Wrapper que retorna dicionário com `exatas` e `near`, compatível com stub."""
    relatorio_exatas = {
        coluna: duplicatas_exatas(df, coluna)
        for coluna in ("link", "title", coluna_texto)
    }
    pares = encontrar_near_duplicates(
        df,
        coluna_texto=coluna_texto,
        threshold=threshold,
        num_perm=num_perm,
        k_shingle=k_shingle,
    )
    return {
        "exatas": relatorio_exatas,
        "near": {
            "threshold": threshold,
            "k_shingle": k_shingle,
            "num_perm": num_perm,
            "n_pares_candidatos": int(len(pares)),
            "pares": pares,
        },
    }
