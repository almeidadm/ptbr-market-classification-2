"""
Llama 3.1 8B zero-shot via vLLM (ADR 009 §D.1, §D.3, §D.4).

Responsabilidades:
- Carregar o modelo quantizado GPTQ-INT4 uma única vez por experimento.
- Renderizar o prompt (um por recorte) substituindo `{text}` pelo artigo.
- Gerar respostas greedy (`temperature=0`, `top_p=1`).
- Parsear a saída para mapear em uma das classes esperadas.
- Registrar taxa off-label (fração de respostas que caíram no fallback).

A inferência propriamente dita requer GPU (L4 confirmada no ciclo
2026-04-24). O parser é testável sem GPU e tem módulo-nível exposto.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    LLAMA_MAX_NEW_TOKENS,
    LLAMA_MODEL_ID,
    LLAMA_QUANTIZATION,
    LLAMA_TEMPERATURE,
    LLAMA_TOP_P,
    MAX_PALAVRAS_LLAMA,
)
from src.modelos.texto import juntar_titulo_texto


# Extração do primeiro termo alfabético (aceita hífen para `nao-mercado`).
_REGEX_TERMO = re.compile(r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*")


def normalizar(texto: str) -> str:
    """Lower + strip + remoção de acentos."""
    decomposto = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(ch for ch in decomposto if not unicodedata.combining(ch))


def parsear_resposta(resposta: str, classes: set[str]) -> str | None:
    """
    Extrai o primeiro termo alfabético (tolerando hífens internos) e
    devolve-o normalizado se estiver no conjunto `classes`; `None` em
    caso de miss — o chamador aplica fallback (ADR 009 §D.4).
    """
    if not resposta:
        return None
    match = _REGEX_TERMO.search(resposta)
    if match is None:
        return None
    termo = normalizar(match.group(0))
    return termo if termo in classes else None


def truncar_por_palavras(texto: str, max_palavras: int = MAX_PALAVRAS_LLAMA) -> str:
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return texto
    return " ".join(palavras[:max_palavras])


def carregar_prompt(caminho: Path) -> str:
    """
    Lê o arquivo markdown de prompt e retorna o template completo
    (incluindo `{text}`). O YAML frontmatter e os separadores markdown
    são preservados — o prompt é consumido na íntegra pela Llama.
    """
    return Path(caminho).read_text(encoding="utf-8")


def renderizar(prompt: str, texto: str) -> str:
    """
    Substitui `{text}` pelo artigo. Usa `str.replace` em vez de
    `str.format` porque o prompt contém chaves literais (YAML, markdown).
    """
    return prompt.replace("{text}", texto)


def preparar_entrada_llama(df: pd.DataFrame) -> pd.Series:
    """Texto concatenado e truncado para consumo pela Llama."""
    textos = df.apply(
        lambda linha: juntar_titulo_texto(linha.get("title"), linha.get("text")),
        axis=1,
    )
    return textos.map(truncar_por_palavras)


def classe_majoritaria_do_fold(y_treino: Iterable[str]) -> str:
    """Fallback off-label (ADR 009 §D.4 + Q5 Leitura única): classe
    mais frequente no treino do fold atual."""
    return pd.Series(list(y_treino)).value_counts().idxmax()


def classificar_llama_zero(
    textos: list[str],
    prompt: str,
    classes: list[str],
    fallback: str,
    motor_vllm,
) -> tuple[list[str], float]:
    """
    Executa inferência em batch via `motor_vllm` e parseia respostas.

    `motor_vllm` é um `vllm.LLM` já inicializado pelo caller (o runner
    reutiliza a mesma instância entre folds para amortizar o custo de
    carga do modelo). Sampling fixo conforme `src.config`.
    """
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=LLAMA_TEMPERATURE,
        top_p=LLAMA_TOP_P,
        max_tokens=LLAMA_MAX_NEW_TOKENS,
    )
    prompts_renderizados = [renderizar(prompt, texto) for texto in textos]
    saidas = motor_vllm.generate(prompts_renderizados, sampling)

    conjunto_classes = set(classes)
    predicoes: list[str] = []
    n_off_label = 0
    for saida in saidas:
        resposta = saida.outputs[0].text if saida.outputs else ""
        pred = parsear_resposta(resposta, conjunto_classes)
        if pred is None:
            n_off_label += 1
            pred = fallback
        predicoes.append(pred)

    off_label_rate = n_off_label / len(predicoes) if predicoes else 0.0
    return predicoes, off_label_rate


def carregar_motor_vllm(
    model_id: str = LLAMA_MODEL_ID,
    quantization: str = LLAMA_QUANTIZATION,
    **kwargs,
):
    """
    Inicializa `vllm.LLM` com os defaults do projeto. Separado em função
    para permitir testes que fiquem isolados da disponibilidade de GPU.
    """
    from vllm import LLM

    return LLM(
        model=model_id,
        quantization=quantization,
        seed=2026,
        **kwargs,
    )
