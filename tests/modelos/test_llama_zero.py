"""Testes para o parser e utilitários de `src.modelos.llama_zero`.

A inferência vLLM em si (`classificar_llama_zero`, `carregar_motor_vllm`)
exige GPU e não é coberta neste arquivo — será exercitada no smoke
validation da Fase 4 ou em corrida real.
"""
from __future__ import annotations

import pandas as pd

from src.modelos.llama_zero import (
    classe_majoritaria_do_fold,
    normalizar,
    parsear_resposta,
    renderizar,
    truncar_por_palavras,
)


CLASSES_OPCAO_7 = {
    "poder", "colunas", "mercado", "esporte", "mundo",
    "cotidiano", "ilustrada", "outros",
}


def test_parse_resposta_exata() -> None:
    assert parsear_resposta("mercado", CLASSES_OPCAO_7) == "mercado"


def test_parse_com_maiusculas_e_pontuacao() -> None:
    assert parsear_resposta("MERCADO.", CLASSES_OPCAO_7) == "mercado"
    assert parsear_resposta("  Mercado  ", CLASSES_OPCAO_7) == "mercado"


def test_parse_com_prefixo_extrai_primeiro_termo_valido() -> None:
    # O parser pega o PRIMEIRO termo alfabético; se não for uma classe
    # válida, retorna None mesmo que haja termo válido depois.
    assert parsear_resposta("a categoria é mercado", CLASSES_OPCAO_7) is None
    assert parsear_resposta("mercado é a resposta", CLASSES_OPCAO_7) == "mercado"


def test_parse_acentos_removidos_em_termo_unico() -> None:
    # O parser pega apenas o primeiro termo alfabético contíguo;
    # acentos são removidos na normalização.
    assert parsear_resposta("São", {"sao"}) == "sao"
    assert parsear_resposta("Opinião", {"opiniao"}) == "opiniao"


def test_parse_resposta_multi_palavra_nao_casa_com_classe_monoliteca() -> None:
    # Classes canônicas do projeto usam nomes sem espaço (`saopaulo`,
    # `paineldoleitor`). Se a Llama responder "São Paulo" com espaço,
    # o primeiro termo é "sao" — não bate com "saopaulo" e cai no fallback.
    # Esse é o comportamento desejado: força o modelo a seguir o prompt.
    assert parsear_resposta("São Paulo", {"saopaulo"}) is None


def test_parse_resposta_invalida_retorna_none() -> None:
    assert parsear_resposta("xyz123", CLASSES_OPCAO_7) is None
    assert parsear_resposta("", CLASSES_OPCAO_7) is None


def test_parse_nao_mercado_com_hifen() -> None:
    classes = {"mercado", "nao-mercado"}
    assert parsear_resposta("nao-mercado", classes) == "nao-mercado"
    assert parsear_resposta("não-mercado", classes) == "nao-mercado"


def test_normalizar_remove_acentos_e_lower() -> None:
    assert normalizar("São Paulo") == "sao paulo"
    assert normalizar("  MERCADO  ") == "mercado"


def test_truncar_por_palavras_respeita_limite() -> None:
    texto = " ".join(["palavra"] * 2000)
    truncado = truncar_por_palavras(texto, max_palavras=1500)
    assert len(truncado.split()) == 1500


def test_truncar_por_palavras_nao_altera_textos_curtos() -> None:
    texto = "notícia breve com poucas palavras"
    assert truncar_por_palavras(texto, max_palavras=1500) == texto


def test_renderizar_substitui_text_mesmo_com_chaves_literais_no_prompt() -> None:
    prompt = "---\nchave: {valor}\n---\n\nTexto:\n{text}\n\nCategoria:"
    resultado = renderizar(prompt, "corpo da notícia")
    assert "corpo da notícia" in resultado
    # Chaves literais do YAML frontmatter são preservadas.
    assert "{valor}" in resultado


def test_classe_majoritaria_do_fold_retorna_mais_frequente() -> None:
    y = ["outros"] * 10 + ["mercado"] * 3 + ["poder"] * 5
    assert classe_majoritaria_do_fold(y) == "outros"
