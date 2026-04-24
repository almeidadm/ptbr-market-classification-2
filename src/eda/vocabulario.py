"""
Tokenização simples, top-k, Jaccard e PSI para as seções §6 e §8 da EDA.

Tokenização: `lower()` seguido de `re.findall(r"\\b\\w+\\b", ...,
flags=re.UNICODE)`. Stopwords PT-BR: preferencialmente `nltk.corpus.
stopwords('portuguese')`; se o NLTK não estiver disponível ou o corpus
não tiver sido baixado, usa uma lista curada mínima embutida (anotada
no código-fonte para auditoria).

`tokens_top_k_tfidf` materializa a §6 (co-ocorrência lexical): agrega
textos por categoria e retorna os `k` termos de maior TF-IDF médio
sobre a matriz categoria × termo. A agregação usa amostra estratificada
determinística (parâmetro `n_amostra_por_categoria`) para manter o custo
controlado em corpus de ~167k notícias sem introduzir viés por ordem.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

_REGEX_TOKEN = re.compile(r"\b\w+\b", flags=re.UNICODE)

# Lista mínima de stopwords PT-BR usada como fallback quando o corpus do
# NLTK não está disponível. É um subconjunto conservador do corpus do
# NLTK; escolhido como fallback para não travar a EDA em ambiente sem
# download do NLTK. A lista canônica deve vir do NLTK sempre que possível.
_STOPWORDS_FALLBACK_PT = frozenset(
    {
        "a", "à", "às", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles",
        "aquilo", "as", "com", "como", "da", "das", "de", "dela", "delas",
        "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas",
        "ele", "eles", "em", "entre", "era", "eram", "essa", "essas", "esse",
        "esses", "esta", "estas", "este", "estes", "estou", "eu", "foi",
        "foram", "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas",
        "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na",
        "não", "nas", "nem", "no", "nos", "nós", "nossa", "nossas", "nosso",
        "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas",
        "pelo", "pelos", "por", "qual", "quando", "que", "quem", "são", "se",
        "seja", "sem", "ser", "será", "seu", "seus", "só", "sobre", "sua",
        "suas", "também", "te", "tem", "tém", "ter", "teu", "teus", "tu",
        "tua", "tuas", "um", "uma", "umas", "uns", "vocês", "você",
    }
)


def _carregar_stopwords_pt() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("portuguese"))
    except Exception:
        return _STOPWORDS_FALLBACK_PT


def tokenizar(texto: str, remover_stopwords: bool = True) -> list[str]:
    """Tokeniza por regex Unicode `\\b\\w+\\b`, opcionalmente filtrando stopwords PT-BR."""
    if not isinstance(texto, str):
        return []
    tokens = _REGEX_TOKEN.findall(texto.lower())
    if not remover_stopwords:
        return tokens
    stopwords_pt = _carregar_stopwords_pt()
    return [token for token in tokens if token not in stopwords_pt]


def tokens_top_k(
    textos: Iterable[str],
    k: int,
    remover_stopwords: bool = True,
) -> list[tuple[str, int]]:
    """Top-k tokens globais, contados sobre todos os textos."""
    contador: Counter[str] = Counter()
    for texto in textos:
        contador.update(tokenizar(texto, remover_stopwords=remover_stopwords))
    return contador.most_common(k)


def jaccard(conjunto_a: Iterable[str], conjunto_b: Iterable[str]) -> float:
    """Jaccard entre dois conjuntos de tokens. Ambos vazios -> 0.0."""
    a = set(conjunto_a)
    b = set(conjunto_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# Padrão de token usado pelo TfidfVectorizer na §6: só letras minúsculas com
# ou sem acentuação portuguesa, mínimo 3 caracteres. Isolado como constante
# porque o mesmo padrão é reutilizado em §8 para consistência de tokenização
# entre as duas seções.
REGEX_TOKEN_TFIDF = r"(?u)\b[a-zá-ú]{3,}\b"


def jaccard_conjuntos(conjunto_a: Iterable[str], conjunto_b: Iterable[str]) -> float:
    """Alias explícito de `jaccard` orientado ao contrato da §6 e §8.

    A §6 chama essa função para comparar top-k tokens entre categorias; a §8
    para trimestres consecutivos. O nome `jaccard_conjuntos` deixa claro no
    chamador que os operandos são tratados como conjuntos (perdendo contagem).
    """
    return jaccard(conjunto_a, conjunto_b)


def _amostra_estratificada_determinista(
    df: pd.DataFrame,
    coluna_categoria: str,
    categorias: list[str],
    n_por_categoria: int,
    seed: int,
) -> pd.DataFrame:
    """Amostra até `n_por_categoria` linhas por categoria com `random_state=seed`.

    Categorias com menos de `n_por_categoria` linhas retornam todas as linhas
    disponíveis. Usa `pd.DataFrame.sample` com `random_state` fixo para
    determinismo, e concatena mantendo a ordem estável de `categorias`.
    """
    partes = []
    for categoria in categorias:
        sub = df.loc[df[coluna_categoria] == categoria]
        if sub.empty:
            continue
        tamanho = min(n_por_categoria, len(sub))
        partes.append(sub.sample(n=tamanho, random_state=seed))
    if not partes:
        return df.iloc[0:0].copy()
    return pd.concat(partes, axis=0, ignore_index=False)


def tokens_top_k_tfidf(
    df: pd.DataFrame,
    coluna_categoria: str,
    coluna_texto: str,
    categorias: Iterable[str],
    k: int = 30,
    n_amostra_por_categoria: int = 2000,
    seed: int = 20260423,
) -> pd.DataFrame:
    """Top-k tokens por categoria segundo TF-IDF médio sobre as linhas da categoria.

    Fluxo:
      1. Amostra estratificada determinística: até `n_amostra_por_categoria`
         linhas por categoria (com `random_state=seed`). Evita carregar o
         corpus inteiro na matriz TF-IDF.
      2. `TfidfVectorizer` com stopwords PT-BR (NLTK com fallback local),
         `token_pattern=REGEX_TOKEN_TFIDF`, `max_df=0.95`, `min_df=2`,
         `ngram_range=(1,1)`.
      3. Para cada categoria, calcula TF-IDF médio por termo sobre as linhas
         daquela categoria e seleciona os `k` maiores.

    Retorna DataFrame em formato longo: `categoria, rank, token, tfidf_mean`
    com `8 * k` linhas (assumindo 8 categorias). `rank` começa em 1.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    categorias = list(categorias)
    amostra = _amostra_estratificada_determinista(
        df=df.loc[df[coluna_texto].notna()],
        coluna_categoria=coluna_categoria,
        categorias=categorias,
        n_por_categoria=n_amostra_por_categoria,
        seed=seed,
    )
    if amostra.empty:
        return pd.DataFrame(columns=["categoria", "rank", "token", "tfidf_mean"])

    stopwords_pt = sorted(_carregar_stopwords_pt())
    vetorizador = TfidfVectorizer(
        lowercase=True,
        stop_words=stopwords_pt,
        token_pattern=REGEX_TOKEN_TFIDF,
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 1),
    )
    matriz = vetorizador.fit_transform(amostra[coluna_texto].astype(str).to_numpy())
    vocabulario = np.asarray(vetorizador.get_feature_names_out())

    categorias_amostra = amostra[coluna_categoria].to_numpy()
    linhas_saida = []
    for categoria in categorias:
        mascara = categorias_amostra == categoria
        if not mascara.any():
            continue
        media = np.asarray(matriz[mascara].mean(axis=0)).ravel()
        # `argsort` estável (mergesort) para desempate determinístico;
        # negamos para obter descendente.
        ordem = np.argsort(-media, kind="mergesort")[:k]
        for rank_idx, posicao in enumerate(ordem, start=1):
            linhas_saida.append(
                {
                    "categoria": categoria,
                    "rank": rank_idx,
                    "token": str(vocabulario[posicao]),
                    "tfidf_mean": float(media[posicao]),
                }
            )

    return pd.DataFrame(linhas_saida, columns=["categoria", "rank", "token", "tfidf_mean"])


def sobreposicao_top_tokens(
    top_tokens: pd.DataFrame,
    categoria_referencia: str,
) -> pd.DataFrame:
    """Calcula Jaccard + termos compartilhados/exclusivos contra uma categoria de referência.

    Espera saída de `tokens_top_k_tfidf`. Retorna DataFrame com uma linha por
    categoria diferente da referência: `categoria_vs, jaccard_top30,
    tokens_compartilhados, tokens_exclusivos_referencia`. `tokens_*` são
    strings separadas por `;` (formato compatível com CSV de uma coluna).
    """
    tokens_por_categoria: dict[str, list[str]] = {}
    for categoria, grupo in top_tokens.groupby("categoria", sort=False):
        tokens_por_categoria[categoria] = (
            grupo.sort_values("rank", kind="mergesort")["token"].tolist()
        )

    if categoria_referencia not in tokens_por_categoria:
        raise KeyError(
            f"categoria de referência '{categoria_referencia}' ausente em top_tokens"
        )

    tokens_ref = set(tokens_por_categoria[categoria_referencia])
    linhas = []
    for categoria, tokens in tokens_por_categoria.items():
        if categoria == categoria_referencia:
            continue
        tokens_outra = set(tokens)
        compartilhados = sorted(tokens_ref & tokens_outra)
        exclusivos_ref = sorted(tokens_ref - tokens_outra)
        linhas.append(
            {
                "categoria_vs": categoria,
                "jaccard_top30": jaccard_conjuntos(tokens_ref, tokens_outra),
                "tokens_compartilhados": ";".join(compartilhados),
                "tokens_exclusivos_referencia": ";".join(exclusivos_ref),
            }
        )
    frame = pd.DataFrame(linhas)
    return frame.sort_values("jaccard_top30", ascending=False, kind="mergesort").reset_index(drop=True)


def psi(
    esperado: dict[str, float],
    observado: dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index entre duas distribuições sobre mesmo vocabulário.

    As entradas são dicionários `token -> proporção`. Tokens fora de uma
    das distribuições contam como `epsilon` para evitar divisão por zero
    (prática comum em aplicações de PSI).
    """
    vocabulario = set(esperado) | set(observado)
    total = 0.0
    for token in vocabulario:
        p_esperado = max(esperado.get(token, 0.0), epsilon)
        p_observado = max(observado.get(token, 0.0), epsilon)
        total += (p_observado - p_esperado) * float(np.log(p_observado / p_esperado))
    return float(total)
