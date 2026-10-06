#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_modelos_para_dataset.py — Baixa os modelos GGUF v1 e publica como Kaggle Dataset.

RODE ESTE SCRIPT DENTRO DE UM NOTEBOOK KAGGLE (nao na sua maquina). Ele:
  1. baixa os modelos (GGUF/encoders/VAE/LoRAs) DIRETO no Kaggle para
     /kaggle/working/comfyui-models/ seguindo o layout da secao 8 do
     arquitetura-recomendada-v1.md;
  2. (opcional, atras do flag PUBLICAR) publica/versiona esse conteudo como um
     Kaggle Dataset, para o kaggle_setup.py anexar depois via extra_model_paths.

Assim os modelos NUNCA passam pela sua maquina local — tudo acontece no Kaggle.

TOKENS: lidos dos SECRETS do Kaggle (Add-ons > Secrets), com fallback para
  variavel de ambiente. Secrets esperados: HF_TOKEN e CIVITAI_TOKEN.

ATENCAO (politica Kaggle): a conta e NOVA e NAO pode gerar/armazenar NSFW. Este
  script lista APENAS variantes limpas. NUNCA adicione modelos NSFW aqui.

COMO USAR (resumo):
  - Primeiro rode com PUBLICAR = False, confira os downloads e os PLACEHOLDERs
    de nome de arquivo (veja comentarios na lista MODELOS).
  - Depois ajuste PUBLICAR = True para criar/versionar o Dataset.
  - A API do Kaggle ja vem disponivel no notebook; pode ser preciso habilitar o
    acesso a API nas configuracoes do notebook.
"""

import json
import os
import subprocess

# --------------------------------------------------------------------------- #
# Config principal (EDITE AQUI)
# --------------------------------------------------------------------------- #
DEST_RAIZ = "/kaggle/working/comfyui-models"   # raiz do Dataset (layout secao 8)
PUBLICAR = False                               # True = cria/versiona o Dataset
DATASET_ID = "marconi2/comfyui-models"         # <user>/<slug> no Kaggle
DATASET_TITULO = "comfyui-models"


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
# LISTA DE MODELOS (EDITE AQUI) — v1 do arquitetura-recomendada-v1.md (secao 6).
#
# Cada item e um dict:
#   origem    : "hf" (HuggingFace) ou "civitai"
#   subpasta  : destino relativo a DEST_RAIZ. Use um de:
#               diffusion_models | text_encoders | vae | loras | checkpoints
#   repo_id   : (hf) "org/repo"
#   arquivo   : (hf) nome EXATO do arquivo no repo
#   civitai_id: (civitai) modelVersionId OU URL completa de download
#   obs       : comentario pt-BR
#
# IMPORTANTE sobre nomes de arquivo: onde o nome EXATO nao e conhecido com
# certeza, o campo "arquivo" vem como "PLACEHOLDER_...". NAO invente nomes:
# ABRA a pagina do repo HF (link no obs), copie o nome exato do .gguf/.safetensors
# e substitua o PLACEHOLDER antes de rodar. O baixar_hf() pula PLACEHOLDERs.
#
# Variantes NSFW NUNCA entram aqui (politica Kaggle — conta nova).
# --------------------------------------------------------------------------- #
MODELOS = [
    # ---------------- IMAGEM: Qwen-Image-2.1 (7B) GGUF Q4_K_M ----------------
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "Abiray/Qwen-Image-2.1-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # ex.: *Q4_K_M.gguf
        "obs": "Qwen-Image-2.1 (7B) Q4_K_M. Confirme o nome exato em "
               "https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF "
               "(evitar Q8_0: ha relato de shape [136] vs [128]).",
    },
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "city96/Qwen-Image-gguf",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # Qwen-Image 20B (Apache)
        "obs": "ALTERNATIVA: Qwen-Image 20B GGUF Q4_K_M (licenca Apache, mais "
               "pesado). Confirme o nome em https://huggingface.co/city96/Qwen-Image-gguf",
    },

    # ---------------- EDICAO: Qwen-Image-Edit-2511 GGUF Q4_K_M ---------------
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "QuantStack/Qwen-Image-Edit-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *2511*Q4_K_M.gguf
        "obs": "Qwen-Image-Edit-2511 Q4_K_M (~13 GB, cabe em 16 GB). Confirme em "
               "https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF "
               "(escolha a variante 2511, NAO a 2509).",
    },
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "calcuis/qwen-image-edit-plus-gguf",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",
        "obs": "ALTERNATIVA do Qwen-Image-Edit (plus/2511). Confirme em "
               "https://huggingface.co/calcuis/qwen-image-edit-plus-gguf",
    },

    # ---------------- VIDEO: Wan 2.2 GGUF (SOMENTE variantes LIMPAS) ---------
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "geceff/Wan2.2-Custom-Models-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # high-noise Q4_K_M LIMPO
        "obs": "Wan 2.2 I2V HIGH-noise Q4_K_M — USAR SO A VARIANTE LIMPA (NAO "
               "NSFW). Confirme em https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF",
    },
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "geceff/Wan2.2-Custom-Models-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # low-noise Q4_K_M LIMPO
        "obs": "Wan 2.2 I2V LOW-noise Q4_K_M — USAR SO A VARIANTE LIMPA (NAO "
               "NSFW). Confirme em https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF",
    },
    {
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "calcuis/wan-gguf",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # ALTERNATIVA: Wan 2.2 5B TI2V
        "obs": "ALTERNATIVA mais leve: Wan 2.2 5B TI2V GGUF (um unico DiT, t2v+i2v). "
               "Confirme em https://huggingface.co/calcuis/wan-gguf",
    },

    # ---------------- TEXT ENCODERS ------------------------------------------
    {
        "origem": "hf",
        "subpasta": "text_encoders",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # qwen_2.5_vl_7b*.safetensors
        "obs": "Text encoder Qwen2.5-VL (imagem/edicao). Confirme o nome em "
               "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI "
               "(pasta split_files/text_encoders). Pode ficar na RAM/CPU.",
    },
    {
        "origem": "hf",
        "subpasta": "text_encoders",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # umt5_xxl*.safetensors
        "obs": "Text encoder UMT5-XXL (Wan). Confirme o nome em "
               "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged "
               "(prefira a variante fp8/scaled para o Dataset nao ficar enorme).",
    },

    # ---------------- VAE -----------------------------------------------------
    {
        "origem": "hf",
        "subpasta": "vae",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # qwen_image_vae.safetensors
        "obs": "VAE do Qwen-Image. Confirme em "
               "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI (split_files/vae).",
    },
    {
        "origem": "hf",
        "subpasta": "vae",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # wan_2.1_vae / wan2.2_vae
        "obs": "VAE do Wan (wan_2.1_vae para 14B; wan2.2_vae para o 5B). Confirme "
               "em https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged.",
    },

    # ---------------- LoRAs de POUCOS PASSOS (Lightning / LightX2V) ----------
    {
        "origem": "hf",
        "subpasta": "loras",
        "repo_id": "QuantStack/Qwen-Image-Edit-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *lightning*4steps*.safetensors
        "obs": "LoRA Lightning 4 passos para Qwen-Image-Edit-2511 (acelera MUITO). "
               "Confirme o repo/arquivo correto da LoRA (pode estar em repo separado). "
               "Para a arquitetura 2.1 (7B) valide a compatibilidade da LoRA.",
    },
    {
        "origem": "hf",
        "subpasta": "loras",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *lightx2v*4steps*.safetensors
        "obs": "LoRA LightX2V de poucos passos para Wan 2.2 (4-12 passos). Confirme "
               "o nome/repo exato antes de baixar.",
    },

    # ---------------- CHECKPOINT fallback (mantem o auto-detect feliz) -------
    # Opcional: o kaggle_setup.py ja baixa o epiCRealism no working como fallback.
    # Se quiser deixa-lo tambem no Dataset, descomente e preencha o civitai_id:
    # {
    #     "origem": "civitai",
    #     "subpasta": "checkpoints",
    #     "civitai_id": "143906",
    #     "arquivo": "epicrealism_naturalSinRC1VAE.safetensors",
    #     "obs": "epiCRealism SD 1.5 (modelVersionId 143906). Leve, roda bem na T4.",
    # },
]


# --------------------------------------------------------------------------- #
# Estrutura de pastas
# --------------------------------------------------------------------------- #
SUBPASTAS = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints"]


def criar_pastas():
    os.makedirs(DEST_RAIZ, exist_ok=True)
    for sub in SUBPASTAS:
        os.makedirs(os.path.join(DEST_RAIZ, sub), exist_ok=True)
    print(">> Pastas criadas em", DEST_RAIZ)


def _tamanho_gb(caminho):
    try:
        return round(os.path.getsize(caminho) / 1024**3, 2)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Download HuggingFace
# --------------------------------------------------------------------------- #
def baixar_hf(m):
    arquivo = m.get("arquivo", "")
    if not arquivo or arquivo.startswith("PLACEHOLDER"):
        print(f">> [PULADO] {m['repo_id']}: nome de arquivo e PLACEHOLDER.")
        print(f">>          Abra a pagina do repo, confirme o nome exato e edite "
              f"a lista MODELOS. ({m.get('obs','')})")
        return

    dest_dir = os.path.join(DEST_RAIZ, m["subpasta"])
    os.makedirs(dest_dir, exist_ok=True)
    destino = os.path.join(dest_dir, os.path.basename(arquivo))
    if os.path.exists(destino):
        print(f">> ja existe: {m['subpasta']}/{os.path.basename(arquivo)} "
              f"({_tamanho_gb(destino)} GB)")
        return

    try:
        from huggingface_hub import hf_hub_download
        print(f">> baixando (HF): {m['repo_id']} :: {arquivo} -> {m['subpasta']}/")
        caminho = hf_hub_download(
            repo_id=m["repo_id"],
            filename=arquivo,
            local_dir=dest_dir,
            token=HF_TOKEN or None,
        )
        # hf_hub_download pode criar subpastas internas do repo; normaliza nome
        if os.path.abspath(caminho) != os.path.abspath(destino) and os.path.exists(caminho):
            try:
                os.replace(caminho, destino)
            except Exception:
                destino = caminho
        print(f">> OK: {os.path.basename(destino)} ({_tamanho_gb(destino)} GB)")
    except Exception as e:
        print(f">> [ERRO] {m['repo_id']} :: {arquivo}: {e}")


# --------------------------------------------------------------------------- #
# Download Civitai
# --------------------------------------------------------------------------- #
def baixar_civitai(m):
    cid = str(m.get("civitai_id", "")).strip()
    if not cid:
        print(f">> [PULADO] Civitai sem civitai_id: {m.get('obs','')}")
        return
    dest_dir = os.path.join(DEST_RAIZ, m["subpasta"])
    os.makedirs(dest_dir, exist_ok=True)
    destino = os.path.join(dest_dir, m.get("arquivo", ""))
    if m.get("arquivo") and os.path.exists(destino):
        print(f">> ja existe: {m['subpasta']}/{m['arquivo']} ({_tamanho_gb(destino)} GB)")
        return

    if cid.startswith("http"):
        url = cid
        sep = "&" if "?" in url else "?"
        if CIVITAI_TOKEN:
            url = f"{url}{sep}token={CIVITAI_TOKEN}"
    else:
        url = f"https://civitai.com/api/download/models/{cid}"
        if CIVITAI_TOKEN:
            url += f"?token={CIVITAI_TOKEN}"

    print(f">> baixando (Civitai): {cid} -> {m['subpasta']}/")
    if m.get("arquivo"):
        run(["wget", "--content-disposition", url, "-O", destino], check=False)
    else:
        # sem nome definido: deixa o --content-disposition nomear o arquivo
        run(["wget", "--content-disposition", url, "-P", dest_dir], check=False)


# --------------------------------------------------------------------------- #
# Metadata + publicacao do Dataset
# --------------------------------------------------------------------------- #
def escrever_metadata():
    meta = {
        "title": DATASET_TITULO,
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],  # ajuste a licenca se precisar
    }
    caminho = os.path.join(DEST_RAIZ, "dataset-metadata.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(">> dataset-metadata.json escrito em", caminho)


def publicar():
    if not PUBLICAR:
        print(">> [info] PUBLICAR=False — download concluido, Dataset NAO publicado.")
        print(">>        Confira os arquivos e os PLACEHOLDERs, depois ponha PUBLICAR=True.")
        return

    # PRIMEIRA VEZ: cria o Dataset. --dir-mode zip evita compactar tudo num zip
    # unico gigante e preserva a estrutura de pastas (bom para modelos de varios GB).
    print(">> Publicando Dataset (1a vez) via Kaggle CLI...")
    run(["kaggle", "datasets", "create", "-p", DEST_RAIZ, "--dir-mode", "zip"], check=False)

    # ATUALIZACOES (nas proximas vezes, NAO use 'create' — use 'version'):
    # run(["kaggle", "datasets", "version", "-p", DEST_RAIZ,
    #      "-m", "novos modelos", "--dir-mode", "zip"], check=False)


def main():
    criar_pastas()
    for m in MODELOS:
        if m.get("origem") == "hf":
            baixar_hf(m)
        elif m.get("origem") == "civitai":
            baixar_civitai(m)
        else:
            print(">> [PULADO] origem desconhecida:", m.get("obs", m))
    escrever_metadata()
    publicar()
    print(">> Concluido.")


if __name__ == "__main__":
    main()
