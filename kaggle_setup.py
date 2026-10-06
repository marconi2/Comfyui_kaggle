#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_setup.py — Setup do ComfyUI no Kaggle (laboratorio de aprendizado).

ESTRATEGIA DE ARMAZENAMENTO (resolve o limite de ~20 GB do /kaggle/working):
- /kaggle/working  -> ~20 GB, apagado ao desligar. Aqui roda o ComfyUI.
- /kaggle/input/<dataset>  -> onde Datasets sao montados: PERSISTENTE, fora dos
  20 GB, somente-leitura, carrega instantaneo. Aqui ficam os MODELOS GRANDES.

Os modelos NAO sao copiados para o working (gastaria os 20 GB). Em vez disso,
o ComfyUI e apontado para o Dataset via extra_model_paths.yaml (read-only), e
uma pasta GRAVAVEL no working guarda modelos baixados na hora (sob demanda).

TOKENS: lidos dos SECRETS do Kaggle (Add-ons > Secrets), NAO colados em celula.
  Secrets esperados: HF_TOKEN, CIVITAI_TOKEN.

COMO USAR NO KAGGLE (uma celula):
    !wget -q https://raw.githubusercontent.com/marconi2/Comfyui_kaggle/main/kaggle_setup.py -O setup.py && python setup.py

Pre-requisitos no notebook: GPU T4 x2 + Internet On; Secrets HF_TOKEN e
CIVITAI_TOKEN criados; e (quando houver) o Dataset de modelos anexado.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
COMFY = "/kaggle/working/ComfyUI"
CKPT_WORKING = COMFY + "/models/checkpoints"          # gravavel (download na hora)
CLOUDFLARED = "/kaggle/working/cloudflared"

# Nome do Dataset de modelos (ajuste se usar outro nome ao criar o Dataset).
# O caminho real dentro de /kaggle/input varia (ex.: /kaggle/input/comfyui-models/
# ou /kaggle/input/datasets/<user>/comfyui-models/comfyui-models/). Por isso o
# script DETECTA automaticamente as pastas que contem subpastas de modelos (ver
# detectar_datasets_base), em vez de depender de um caminho fixo.
DATASET_DIR = ""   # 1o Dataset detectado (compatibilidade); preenchido em runtime
DATASETS = []      # lista COMPLETA de bases detectadas; preenchida em runtime

# Subpastas de modelos reconhecidas (ordem usada ao escrever o YAML). As chaves
# 'diffusion_models' e 'text_encoders' sao o que faz os GGUF de Qwen/Wan e os
# text encoders aparecerem nos nos (UnetLoaderGGUF / CLIPLoader) — ver secao 8
# do arquitetura-recomendada-v1.md.
SUBPASTAS_MODELOS = [
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "loras",
    "vae",
    "clip",
    "unet",
    "controlnet",
    "upscale_models",
]

# Subconjunto "gatilho": basta UMA destas existir numa pasta para considera-la
# um Dataset de modelos durante a deteccao.
SUBPASTAS_GATILHO = {
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "loras",
    "vae",
    "unet",
}


# --------------------------------------------------------------------------- #
# Tokens via Secrets do Kaggle (com fallback para variavel de ambiente)
# --------------------------------------------------------------------------- #
def ler_secret(nome: str) -> str:
    # 1) tenta o Secrets do Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(nome)
    except Exception:
        pass
    # 2) fallback: variavel de ambiente (caso rode fora do Kaggle)
    return os.environ.get(nome, "")


HF_TOKEN = ler_secret("HF_TOKEN")
CIVITAI_TOKEN = ler_secret("CIVITAI_TOKEN")


def run(cmd, **kw):
    print(">>", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


# --------------------------------------------------------------------------- #
# Instalacao do ComfyUI e do Manager
# --------------------------------------------------------------------------- #
def instalar_comfyui():
    if not os.path.exists(COMFY):
        run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", COMFY], check=True)
        run(["pip", "install", "-r", COMFY + "/requirements.txt"], check=True)
        print(">> ComfyUI instalado")
    else:
        print(">> ComfyUI ja existe")


def instalar_manager():
    mgr = COMFY + "/custom_nodes/comfyui-manager"
    if not os.path.exists(mgr):
        run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI-Manager", mgr], check=True)
        run(["pip", "install", "-r", mgr + "/requirements.txt"], check=True)
        print(">> Manager instalado")
    else:
        print(">> Manager ja existe")


# --------------------------------------------------------------------------- #
# Custom nodes extras (fixos) — reinstalados a cada sessao, pois o working some.
# Para ADICIONAR um node: coloque a URL do repositorio git na lista CUSTOM_NODES.
# O setup faz git clone + instala requirements.txt + roda install.py (se houver).
# --------------------------------------------------------------------------- #
CUSTOM_NODES = [
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack",
    # carregar modelos quantizados GGUF (Qwen/Wan DiT, text encoders, VAE).
    # USAMOS O FORK leejet (nao o city96): o leejet suporta o Qwen-Image-2.1
    # (int8 convrot / input_act) nas versoes novas do ComfyUI; o city96 da o
    # erro "forward_ggml_cast_weights() got an unexpected keyword 'input_act'".
    "https://github.com/leejet/ComfyUI-GGUF",
    # NOTA: este repo esta marcado para ARQUIVAMENTO em 30/09/2026 — ainda
    # funcional e instalavel, mas sem suporte ativo (ver arquitetura v1).
    "https://github.com/pollockjj/ComfyUI-MultiGPU",
    # wrapper de video Wan 2.2 (integra nos MultiGPU dedicados)
    "https://github.com/kijai/ComfyUI-WanVideoWrapper",
    # adicione outros aqui, ex.:
    # "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
]


def instalar_custom_nodes():
    cn_dir = COMFY + "/custom_nodes"
    for url in CUSTOM_NODES:
        nome = url.rstrip("/").split("/")[-1]
        destino = os.path.join(cn_dir, nome)
        if os.path.exists(destino):
            print(f">> node ja existe: {nome}")
            continue
        print(f">> instalando node: {nome}")
        run(["git", "clone", url, destino], check=True)
        req = os.path.join(destino, "requirements.txt")
        if os.path.exists(req):
            run(["pip", "install", "-r", req], check=False)
        inst = os.path.join(destino, "install.py")
        if os.path.exists(inst):
            run(["python", inst], check=False)
        print(f">> node pronto: {nome}")


# --------------------------------------------------------------------------- #
# Apontar o ComfyUI para os modelos do Dataset (SEM copiar — read-only)
# --------------------------------------------------------------------------- #
def detectar_datasets_base():
    """Detecta TODAS as bases de Dataset de modelos sob /kaggle/input.

    Antes o script parava na PRIMEIRA pasta 'checkpoints' encontrada. Agora
    suporta VARIOS Datasets anexados: varremos /kaggle/input em profundidade 2
    (as entradas diretas /kaggle/input/<x> E as subpastas /kaggle/input/<x>/<y>,
    porque o Kaggle as vezes aninha o Dataset uma pasta a mais) e aceitamos como
    base QUALQUER pasta que contenha ao menos UMA subpasta de modelos reconhecida
    (checkpoints, diffusion_models, text_encoders, loras, vae ou unet).

    Retorna a LISTA de caminhos base detectados (pode ser vazia). Nunca quebra:
    tudo e protegido por isdir/try-except.
    """
    raiz = "/kaggle/input"
    if not os.path.isdir(raiz):
        return []

    # 1) monta a lista de candidatos varrendo ate PROFUNDIDADE 5. O Kaggle aninha
    #    o Dataset de formas variadas (ex.: /kaggle/input/<slug>/ OU
    #    /kaggle/input/datasets/<user>/<slug>/), entao em vez de fixar 1-2 niveis
    #    descemos ate achar a pasta que contem as subpastas de modelo.
    PROF_MAX = 5
    candidatos = []
    try:
        for atual, dirs, _ in os.walk(raiz):
            prof = atual.rstrip("/").count("/") - raiz.rstrip("/").count("/")
            if prof >= PROF_MAX:
                dirs[:] = []  # nao desce mais
                continue
            if atual != raiz:
                candidatos.append(atual)
    except Exception:
        return []

    # 2) qualifica cada candidato: contem alguma subpasta-gatilho?
    def eh_dataset(base):
        try:
            for nome in SUBPASTAS_GATILHO:
                if os.path.isdir(os.path.join(base, nome)):
                    return True
        except Exception:
            pass
        return False

    qualificados = [c for c in candidatos if eh_dataset(c)]

    # 3) deduplicar: se um filho qualificou, nao incluir o pai que so qualificou
    #    por conter esse filho (evita montar o mesmo Dataset duas vezes).
    datasets = []
    for base in qualificados:
        tem_filho_qualificado = any(
            outro != base and outro.startswith(base.rstrip("/") + "/")
            for outro in qualificados
        )
        if tem_filho_qualificado:
            # o pai so entra se ELE PROPRIO tem subpasta de modelo direta
            # alem do filho — checagem simples: pular pais que apenas aninham.
            continue
        if base not in datasets:
            datasets.append(base)

    print(f">> Dataset(s) de modelos detectado(s): {len(datasets)}")
    for d in datasets:
        print("    -", d)
    return datasets


def _bloco_yaml_dataset(chave, base):
    """Monta UM bloco top-level do extra_model_paths.yaml para uma base.

    Escreve a linha de uma subpasta apenas se ela existir naquela base (o
    ComfyUI tolera caminhos ausentes, mas assim o YAML fica limpo e previsivel).
    """
    linhas = [f"{chave}:", f"    base_path: {base}"]
    for nome in SUBPASTAS_MODELOS:
        if os.path.isdir(os.path.join(base, nome)):
            linhas.append(f"    {nome}: {nome}")
    return "\n".join(linhas) + "\n"


def configurar_dataset():
    """Cria extra_model_paths.yaml apontando para os Datasets, se existirem.

    O ComfyUI le modelos de VARIAS pastas: as do working (gravaveis, download na
    hora) E as listadas no extra_model_paths.yaml (os Datasets, read-only). Agora
    suportamos MULTIPLOS Datasets anexados: escrevemos UM bloco top-level por
    Dataset (chaves unicas kaggle_dataset_0, kaggle_dataset_1, ...), cada um
    mapeando as subpastas de modelos que existirem naquele Dataset. O ComfyUI
    mescla as varias raizes automaticamente.
    """
    global DATASET_DIR, DATASETS
    DATASETS = detectar_datasets_base()
    if not DATASETS:
        print(">> [info] Nenhum Dataset de modelos encontrado em /kaggle/input.")
        print(">>        Anexe o Dataset de modelos ao notebook (+ Add Input).")
        DATASET_DIR = ""
        return

    # 1o Dataset vira o DATASET_DIR (compatibilidade com baixar_epicrealism_working)
    DATASET_DIR = DATASETS[0]

    blocos = []
    for i, base in enumerate(DATASETS):
        blocos.append(_bloco_yaml_dataset(f"kaggle_dataset_{i}", base))
    yaml = "\n".join(blocos)

    destino = COMFY + "/extra_model_paths.yaml"
    with open(destino, "w", encoding="utf-8") as f:
        f.write(yaml)
    print(">> Dataset(s) conectado(s) via extra_model_paths.yaml")

    # lista o que tem em cada Dataset (ajuda a conferir)
    for i, base in enumerate(DATASETS):
        print(f">> Dataset {i}: {base}")
        for pasta in ("checkpoints", "diffusion_models"):
            caminho = os.path.join(base, pasta)
            if os.path.isdir(caminho):
                print(f"   {pasta}:")
                try:
                    for f in os.listdir(caminho):
                        print("    -", f)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Download sob demanda (para o WORKING — modelos pequenos / testes rapidos)
# --------------------------------------------------------------------------- #
def baixar_epicrealism_working():
    """Baixa o epiCRealism para o working SOMENTE se ele nao estiver no Dataset.

    Se o modelo ja veio no Dataset (detectado em configurar_dataset), nao baixa
    nada — o ComfyUI le direto do Dataset. So baixa no working como fallback.
    """
    nome = "epicrealism_naturalSinRC1VAE.safetensors"
    # ja esta no Dataset? entao nao precisa baixar
    if DATASET_DIR and os.path.exists(os.path.join(DATASET_DIR, "checkpoints", nome)):
        print(">> epiCRealism ja esta no Dataset — nao baixa no working")
        return
    destino = CKPT_WORKING + "/" + nome
    if os.path.exists(destino):
        print(">> epiCRealism ja existe no working")
        return
    if not CIVITAI_TOKEN:
        print(">> [PULADO] epiCRealism: Secret CIVITAI_TOKEN ausente e nao esta no Dataset.")
        return
    os.makedirs(CKPT_WORKING, exist_ok=True)
    url = "https://civitai.com/api/download/models/143906?token=" + CIVITAI_TOKEN
    print(">> Baixando epiCRealism (~2 GB) para o working...")
    run(["wget", "--content-disposition", url, "-O", destino], check=True)
    if os.path.exists(destino):
        print(">> epiCRealism:", round(os.path.getsize(destino) / 1024**3, 2), "GB")


# --------------------------------------------------------------------------- #
# Servidor + tunel
# --------------------------------------------------------------------------- #
def baixar_cloudflared():
    if not os.path.exists(CLOUDFLARED):
        run(["wget", "-q",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
             "-O", CLOUDFLARED], check=True)
        run(["chmod", "+x", CLOUDFLARED], check=True)
        print(">> cloudflared pronto")
    return CLOUDFLARED


def subir_servidor_e_tunel(cf):
    run("pkill -f main.py")
    run("pkill -f cloudflared")
    time.sleep(3)

    os.chdir(COMFY)
    comfy = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _log():
        for l in comfy.stdout:
            print("[comfy]", l, end="")
            if "Starting server" in l:
                break
    threading.Thread(target=_log, daemon=True).start()
    print(">> Subindo ComfyUI (aguarde ~45s)...")
    time.sleep(45)

    tun = subprocess.Popen([cf, "tunnel", "--url", "http://127.0.0.1:8188"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for l in tun.stdout:
        print("[tunel]", l, end="")
        m = re.search(r"https://[-\w]+\.trycloudflare\.com", l)
        if m:
            print("\n\n>>> ABRA NO NAVEGADOR:", m.group(0), "\n")
            break
    comfy.wait()


def main():
    instalar_comfyui()
    instalar_manager()
    instalar_custom_nodes()     # nodes fixos (Inspire Pack etc.) — reinstala sempre
    configurar_dataset()        # aponta para o Dataset (modelos grandes), se anexado
    baixar_epicrealism_working()  # modelo leve para aprender agora
    cf = baixar_cloudflared()
    subir_servidor_e_tunel(cf)


if __name__ == "__main__":
    main()
