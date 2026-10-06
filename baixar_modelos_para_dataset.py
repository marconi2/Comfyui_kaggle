#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_modelos_para_dataset.py — Baixa os modelos GGUF e publica como Kaggle Dataset.

>>> RODE ESTE SCRIPT DENTRO DE UM NOTEBOOK KAGGLE (nao na sua maquina). <<<

PROBLEMA QUE ESTE SCRIPT RESOLVE:
  O /kaggle/working tem SO ~20 GB. Baixar TODOS os modelos de uma vez (~60 GB)
  estoura o disco. E o 'kaggle datasets version' SUBSTITUI o conteudo pela pasta
  enviada (nao e append incremental facil). Entao a estrategia aqui e:

    -> UM DATASET PEQUENO POR GRUPO de modelos.

  Cada grupo cabe folgado nos 20 GB. Voce roda o script UMA VEZ POR GRUPO
  (muda GRUPO_ATUAL no topo), ele baixa so aquele grupo, publica como Dataset
  'comfyui-<grupo>', e (opcional) limpa o working para o proximo grupo.

  No notebook do ComfyUI voce anexa TODOS os Datasets. O kaggle_setup.py ja
  detecta multiplos Datasets e junta tudo no extra_model_paths.yaml.

FLUXO RECOMENDADO (passo a passo):
  1. GRUPO_ATUAL = "qwen-image"; PUBLICAR = False  -> roda, confere os downloads.
  2. PUBLICAR = True                               -> roda de novo, publica o Dataset.
  3. LIMPAR_APOS = True (ou limpe manualmente)     -> esvazia o working.
  4. Troca GRUPO_ATUAL = "qwen-edit", repete 1-3.
  5. Troca GRUPO_ATUAL = "wan", repete 1-3.
  Depois anexe comfyui-qwen-image + comfyui-qwen-edit + comfyui-wan ao notebook
  do ComfyUI.

TOKENS: lidos dos SECRETS do Kaggle (Add-ons > Secrets), com fallback para
  variavel de ambiente. Secrets esperados: HF_TOKEN e CIVITAI_TOKEN.

ATENCAO (politica Kaggle): a conta e NOVA e NAO pode gerar/armazenar NSFW. Este
  script lista APENAS variantes limpas. NUNCA adicione modelos NSFW aqui.
"""

import json
import os
import shutil
import subprocess

# --------------------------------------------------------------------------- #
# >>> CONFIG PRINCIPAL (EDITE AQUI A CADA RODADA) <<<
# --------------------------------------------------------------------------- #
# Qual grupo baixar/publicar AGORA. Rode uma vez por grupo. Veja os grupos
# disponiveis na lista MODELOS abaixo (campo "grupo").
GRUPO_ATUAL = "qwen-image"      # "qwen-image" | "qwen-edit" | "wan"

PUBLICAR = False                # False = so baixa e confere. True = publica o Dataset.
LIMPAR_APOS = False             # True = apaga o working do grupo apos publicar.

# Seu usuario do KAGGLE (NAO e o do GitHub! marconi2 e o GitHub). Vai no id do
# Dataset. Se ficar "", o script usa o KAGGLE_USERNAME do ambiente / kaggle.json.
USUARIO_KAGGLE = "letroprintdigital"   # usuario real da conta Kaggle

# Teto de seguranca: se o grupo passar disto, o script AVISA antes de estourar
# os 20 GB do working. Deixe com folga (working tem ~20 GB no total).
TETO_GB_AVISO = 18.0

# Raiz onde o grupo e montado antes de virar Dataset.
WORK_RAIZ = "/kaggle/working"


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
# LISTA DE MODELOS (EDITE OS NOMES "arquivo" ANTES DE RODAR) — secao 6 do
# arquitetura-recomendada-v1.md.
#
# Cada item e um dict:
#   grupo     : agrupa os modelos num mesmo Dataset. Rode um grupo por vez.
#   origem    : "hf" (HuggingFace) ou "civitai"
#   subpasta  : destino relativo a raiz do grupo. Use um de:
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
    # ======================= GRUPO: qwen-image =============================== #
    # Qwen-Image-2.1 (7B) GGUF Q4_K_M + encoder + VAE. ~15 GB no total.
    {
        "grupo": "qwen-image",
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "Abiray/Qwen-Image-2.1-GGUF",
        "arquivo": "qwen_image_2.1_Q6_K.gguf",  # confirmado na API do HF (2026-09-20)
        "obs": "Qwen-Image-2.1 (7B) Q6_K (~6 GB). Mais qualidade que o Q4_K_M, "
               "porem mais lento na T4. Se ficar lento, troque para "
               "'qwen_image_2.1_Q4_K_M.gguf' (mesmo repo).",
    },
    {
        "grupo": "qwen-image",
        "origem": "hf",
        "subpasta": "text_encoders",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "arquivo": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "obs": "Text encoder Qwen2.5-VL fp8_scaled (~9,4 GB) — roda na T4. NAO usar "
               "a variante 'nvfp4' (so GPU Blackwell/RTX 50xx). Pode ficar na RAM/CPU. "
               "Confirmado na API do HF.",
    },
    {
        "grupo": "qwen-image",
        "origem": "hf",
        "subpasta": "vae",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "arquivo": "split_files/vae/qwen_image_vae.safetensors",  # ~250 MB, confirmado na API do HF
        "obs": "VAE do Qwen-Image (~250 MB). Confirmado na API do HF.",
    },

    # ======================= GRUPO: qwen-edit =============================== #
    # Qwen-Image-Edit-2511 GGUF Q4_K_M + LoRA Lightning. O encoder/VAE Qwen do
    # grupo qwen-image servem para a edicao tambem (anexe os dois Datasets).
    {
        "grupo": "qwen-edit",
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "QuantStack/Qwen-Image-Edit-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *2511*Q4_K_M.gguf
        "obs": "Qwen-Image-Edit-2511 Q4_K_M (~13 GB, cabe em 16 GB). Confirme em "
               "https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF "
               "(escolha a variante 2511, NAO a 2509).",
    },
    {
        "grupo": "qwen-edit",
        "origem": "hf",
        "subpasta": "loras",
        "repo_id": "PLACEHOLDER_confirmar_repo_da_LoRA",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *lightning*4steps*.safetensors
        "obs": "LoRA Lightning 4 passos para Qwen-Image-Edit-2511 (acelera MUITO). "
               "Confirme o repo/arquivo correto da LoRA (costuma estar em repo "
               "separado, ex. lightx2v ou do proprio autor do Edit).",
    },

    # ========================== GRUPO: wan ================================== #
    # Wan 2.2 GGUF (SOMENTE variantes LIMPAS) + encoder + VAE + LoRA.
    # ATENCAO ao total: high+low noise 14B + UMT5 pode passar dos 20 GB. Se
    # estourar, use o 5B TI2V (um DiT so) OU separe em dois grupos (wan-dit e
    # wan-encoder). Rode com PUBLICAR=False primeiro e veja o aviso de tamanho.
    {
        "grupo": "wan",
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "geceff/Wan2.2-Custom-Models-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # high-noise Q4_K_M LIMPO
        "obs": "Wan 2.2 I2V HIGH-noise Q4_K_M — USAR SO A VARIANTE LIMPA (NAO "
               "NSFW). Confirme em https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF",
    },
    {
        "grupo": "wan",
        "origem": "hf",
        "subpasta": "diffusion_models",
        "repo_id": "geceff/Wan2.2-Custom-Models-GGUF",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # low-noise Q4_K_M LIMPO
        "obs": "Wan 2.2 I2V LOW-noise Q4_K_M — USAR SO A VARIANTE LIMPA (NAO "
               "NSFW). Confirme em https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF",
    },
    {
        "grupo": "wan",
        "origem": "hf",
        "subpasta": "text_encoders",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # umt5_xxl*fp8*.safetensors
        "obs": "Text encoder UMT5-XXL (Wan). Confirme o nome em "
               "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged "
               "(prefira a variante fp8/scaled para o Dataset nao ficar enorme).",
    },
    {
        "grupo": "wan",
        "origem": "hf",
        "subpasta": "vae",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # wan_2.1_vae / wan2.2_vae
        "obs": "VAE do Wan (wan_2.1_vae para 14B; wan2.2_vae para o 5B). Confirme "
               "em https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged.",
    },
    {
        "grupo": "wan",
        "origem": "hf",
        "subpasta": "loras",
        "repo_id": "PLACEHOLDER_confirmar_repo_da_LoRA",
        "arquivo": "PLACEHOLDER_confirmar_na_pagina_do_repo_HF",  # *lightx2v*4steps*.safetensors
        "obs": "LoRA LightX2V de poucos passos para Wan 2.2 (4-12 passos). Confirme "
               "o nome/repo exato antes de baixar.",
    },
]


# --------------------------------------------------------------------------- #
# Derivados do grupo atual
# --------------------------------------------------------------------------- #
def dest_raiz_grupo() -> str:
    """Pasta no working onde o grupo atual e montado antes de virar Dataset."""
    return os.path.join(WORK_RAIZ, f"comfyui-{GRUPO_ATUAL}")


def usuario_kaggle() -> str:
    """Usuario dono do Dataset. Usa USUARIO_KAGGLE se preenchido; senao pega do
    ambiente (KAGGLE_USERNAME) definido ao autenticar a API — assim o id SEMPRE
    bate com quem autenticou, evitando 'Invalid Owner Id'."""
    if USUARIO_KAGGLE.strip():
        return USUARIO_KAGGLE.strip()
    env = os.environ.get("KAGGLE_USERNAME", "").strip()
    if env:
        return env
    # ultimo recurso: tenta ler o ~/.kaggle/kaggle.json
    try:
        caminho = os.path.expanduser("~/.kaggle/kaggle.json")
        with open(caminho, encoding="utf-8") as f:
            return json.load(f).get("username", "").strip()
    except Exception:
        return ""


def dataset_id() -> str:
    u = usuario_kaggle()
    if not u:
        print(">> [AVISO] usuario do Kaggle vazio! Defina USUARIO_KAGGLE no script "
              "ou KAGGLE_USERNAME no ambiente, senao a publicacao falha "
              "('Invalid Owner Id').")
    return f"{u}/comfyui-{GRUPO_ATUAL}"


def modelos_do_grupo():
    return [m for m in MODELOS if m.get("grupo") == GRUPO_ATUAL]


# --------------------------------------------------------------------------- #
# Estrutura de pastas
# --------------------------------------------------------------------------- #
SUBPASTAS = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints"]


def criar_pastas():
    raiz = dest_raiz_grupo()
    os.makedirs(raiz, exist_ok=True)
    for sub in SUBPASTAS:
        os.makedirs(os.path.join(raiz, sub), exist_ok=True)
    print(f">> Pastas criadas em {raiz}")


def _tamanho_gb(caminho):
    try:
        return round(os.path.getsize(caminho) / 1024**3, 2)
    except Exception:
        return 0.0


def tamanho_total_gb(raiz) -> float:
    total = 0.0
    for pasta, _, arquivos in os.walk(raiz):
        for a in arquivos:
            total += _tamanho_gb(os.path.join(pasta, a))
    return round(total, 2)


# --------------------------------------------------------------------------- #
# Download HuggingFace
# --------------------------------------------------------------------------- #
def baixar_hf(m):
    arquivo = m.get("arquivo", "")
    repo_id = m.get("repo_id", "")
    if (not arquivo or arquivo.startswith("PLACEHOLDER")
            or not repo_id or repo_id.startswith("PLACEHOLDER")):
        print(f">> [PULADO] {repo_id}: nome de arquivo/repo e PLACEHOLDER.")
        print(f">>          Abra a pagina do repo, confirme o nome exato e edite "
              f"a lista MODELOS. ({m.get('obs','')})")
        return

    dest_dir = os.path.join(dest_raiz_grupo(), m["subpasta"])
    os.makedirs(dest_dir, exist_ok=True)
    # 'arquivo' pode ter caminho interno do repo (ex.: split_files/vae/x.safetensors).
    # O ComfyUI precisa do arquivo ACHATADO direto na subpasta (text_encoders/x,
    # vae/x, ...), entao o destino usa SO o nome base.
    nome_base = os.path.basename(arquivo)
    destino = os.path.join(dest_dir, nome_base)
    if os.path.exists(destino):
        print(f">> ja existe: {m['subpasta']}/{nome_base} ({_tamanho_gb(destino)} GB)")
        return

    try:
        from huggingface_hub import hf_hub_download
        print(f">> baixando (HF): {repo_id} :: {arquivo} -> {m['subpasta']}/{nome_base}")
        # local_dir_use_symlinks=False -> baixa o arquivo REAL (nao um symlink para
        # o cache), essencial para o upload do Dataset enxergar os bytes.
        caminho = hf_hub_download(
            repo_id=repo_id,
            filename=arquivo,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            token=HF_TOKEN or None,
        )
        # Se o HF recriou subpastas internas (split_files/...), ACHATA: move o
        # arquivo para dest_dir/nome_base e remove as pastas aninhadas vazias.
        caminho = os.path.realpath(caminho)
        if os.path.abspath(caminho) != os.path.abspath(destino):
            shutil.move(caminho, destino)
            # remove a arvore aninhada (ex.: dest_dir/split_files) se ficou vazia
            if os.sep in arquivo or "/" in arquivo:
                topo = os.path.join(dest_dir, arquivo.replace("/", os.sep).split(os.sep)[0])
                if os.path.isdir(topo):
                    shutil.rmtree(topo, ignore_errors=True)
        print(f">> OK: {m['subpasta']}/{nome_base} ({_tamanho_gb(destino)} GB)")
    except TypeError:
        # Versoes novas do huggingface_hub removeram local_dir_use_symlinks.
        # Refaz sem o argumento (o default ja baixa arquivo real nessas versoes).
        try:
            from huggingface_hub import hf_hub_download
            caminho = hf_hub_download(
                repo_id=repo_id, filename=arquivo, local_dir=dest_dir,
                token=HF_TOKEN or None,
            )
            caminho = os.path.realpath(caminho)
            if os.path.abspath(caminho) != os.path.abspath(destino):
                shutil.move(caminho, destino)
                if os.sep in arquivo or "/" in arquivo:
                    topo = os.path.join(dest_dir, arquivo.replace("/", os.sep).split(os.sep)[0])
                    if os.path.isdir(topo):
                        shutil.rmtree(topo, ignore_errors=True)
            print(f">> OK: {m['subpasta']}/{nome_base} ({_tamanho_gb(destino)} GB)")
        except Exception as e:
            print(f">> [ERRO] {repo_id} :: {arquivo}: {e}")
    except Exception as e:
        print(f">> [ERRO] {repo_id} :: {arquivo}: {e}")


# --------------------------------------------------------------------------- #
# Download Civitai
# --------------------------------------------------------------------------- #
def baixar_civitai(m):
    cid = str(m.get("civitai_id", "")).strip()
    if not cid:
        print(f">> [PULADO] Civitai sem civitai_id: {m.get('obs','')}")
        return
    dest_dir = os.path.join(dest_raiz_grupo(), m["subpasta"])
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
        run(["wget", "--content-disposition", url, "-P", dest_dir], check=False)


# --------------------------------------------------------------------------- #
# Metadata + publicacao do Dataset
# --------------------------------------------------------------------------- #
def limpar_cache_hf():
    """Remove as pastas .cache/huggingface deixadas pelo hf_hub_download.

    Elas so tem locks/metadados (0 bytes uteis) e NAO devem ir para o Dataset —
    poluem a estrutura. Varre a raiz do grupo e apaga toda pasta '.cache'.
    """
    raiz = dest_raiz_grupo()
    removidas = 0
    for pasta, dirs, _ in os.walk(raiz):
        if ".cache" in dirs:
            shutil.rmtree(os.path.join(pasta, ".cache"), ignore_errors=True)
            removidas += 1
    if removidas:
        print(f">> limpei {removidas} pasta(s) .cache/huggingface antes de publicar")


def escrever_metadata():
    meta = {
        "title": f"comfyui-{GRUPO_ATUAL}",
        "id": dataset_id(),
        "licenses": [{"name": "CC0-1.0"}],  # ajuste a licenca se precisar
    }
    caminho = os.path.join(dest_raiz_grupo(), "dataset-metadata.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(">> dataset-metadata.json escrito em", caminho)


def dataset_existe() -> bool:
    """Checa se o Dataset ja existe (para decidir entre 'create' e 'version')."""
    r = run(["kaggle", "datasets", "status", dataset_id()],
            check=False, capture_output=True, text=True)
    saida = (r.stdout or "") + (r.stderr or "")
    # 'status' retorna erro/'not found' se o Dataset ainda nao existe.
    return r.returncode == 0 and "not found" not in saida.lower()


def publicar():
    raiz = dest_raiz_grupo()
    if not PUBLICAR:
        print(">> [info] PUBLICAR=False — download concluido, Dataset NAO publicado.")
        print(">>        Confira os arquivos e os PLACEHOLDERs, depois PUBLICAR=True.")
        return

    limpar_cache_hf()
    escrever_metadata()
    # --dir-mode zip: empacota cada subpasta para o upload (necessario — 'skip'
    # IGNORA as subpastas e sobe um Dataset vazio). Ao ANEXAR o Dataset a um
    # notebook, o Kaggle descompacta e os arquivos ficam acessiveis como
    # diretorio (diffusion_models/, text_encoders/, vae/...), que e o que o
    # ComfyUI e o auto-detect do kaggle_setup.py esperam.
    if dataset_existe():
        print(f">> Dataset {dataset_id()} ja existe — enviando NOVA VERSAO...")
        run(["kaggle", "datasets", "version", "-p", raiz,
             "-m", f"atualiza {GRUPO_ATUAL}", "--dir-mode", "zip"], check=False)
    else:
        print(f">> Publicando Dataset {dataset_id()} (1a vez)...")
        run(["kaggle", "datasets", "create", "-p", raiz, "--dir-mode", "zip"],
            check=False)
    print(">> OBS: apos publicar, ANEXE o Dataset ao notebook do ComfyUI "
          "(+ Add Input) para o kaggle_setup.py detecta-lo.")


def limpar():
    if not LIMPAR_APOS:
        return
    raiz = dest_raiz_grupo()
    print(f">> LIMPAR_APOS=True — apagando {raiz} para liberar o working...")
    shutil.rmtree(raiz, ignore_errors=True)
    print(">> working liberado.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    grupo_itens = modelos_do_grupo()
    if not grupo_itens:
        print(f">> [ERRO] GRUPO_ATUAL={GRUPO_ATUAL!r} nao tem itens na lista MODELOS.")
        grupos = sorted({m.get("grupo", "?") for m in MODELOS})
        print(f">>        Grupos disponiveis: {grupos}")
        return

    print(f">> ===== GRUPO: {GRUPO_ATUAL}  ({len(grupo_itens)} itens) =====")
    criar_pastas()

    for m in grupo_itens:
        if m.get("origem") == "hf":
            baixar_hf(m)
        elif m.get("origem") == "civitai":
            baixar_civitai(m)
        else:
            print(">> [PULADO] origem desconhecida:", m.get("obs", m))

    total = tamanho_total_gb(dest_raiz_grupo())
    print(f"\n>> Tamanho do grupo '{GRUPO_ATUAL}' no working: {total} GB")
    if total > TETO_GB_AVISO:
        print(f">> [AVISO] {total} GB passou do teto de {TETO_GB_AVISO} GB! O working")
        print(">>         tem ~20 GB. Considere dividir este grupo em dois Datasets")
        print(">>         (ex.: um so para o text encoder) ou usar um quant menor.")

    publicar()
    limpar()
    print(">> Concluido.")


if __name__ == "__main__":
    main()
