# Guia — Modelos & Kaggle Dataset (v1)

Este guia explica como montar e manter o **Kaggle Dataset de modelos** que
alimenta o ComfyUI rodando no Kaggle com **2× T4**.

O ambiente v1 roda **tudo local/open-source**, sem API paga:

- **Qwen-Image-2.1** (7B, GGUF) — geração de imagem (texto→imagem e imagem→imagem).
- **Qwen-Image-Edit-2511** (GGUF) — edição de imagem.
- **Wan 2.2** (GGUF) — vídeo.

A análise completa (hardware T4, VRAM, velocidade, licenças e decisão de não usar
MiniMax) está em [`.agents/tasks/arquitetura-recomendada-v1.md`](.agents/tasks/arquitetura-recomendada-v1.md).
Este guia é a parte prática: **onde cada arquivo mora e como publicar/atualizar o Dataset**.

> Os modelos são baixados **direto no Kaggle** (nunca pela sua máquina) pelo script
> `baixar_modelos_para_dataset.py`, e ficam num Kaggle Dataset persistente, fora
> dos ~20 GB voláteis do `/kaggle/working`.

---

## Layout de pastas do Dataset

Estrutura (secão 8 do arch doc). A raiz precisa conter `checkpoints/` para a
auto-detecção do `kaggle_setup.py` ficar feliz:

```
comfyui-models/
├── checkpoints/          # SD 1.5 / fallback (ex.: epiCRealism)
├── diffusion_models/     # DiTs em GGUF (Qwen-Image, Qwen-Image-Edit, Wan 2.2)
├── text_encoders/        # Qwen2.5-VL (imagem/edição) e UMT5 (Wan)
├── vae/                  # qwen_image_vae, wan vae
├── loras/                # LoRAs Lightning / LightX2V (poucos passos)
├── clip/                 # reservado (CLIP padrão, se algum workflow pedir)
├── controlnet/
├── unet/                 # alternativa a diffusion_models p/ ComfyUI-GGUF
└── upscale_models/
```

Qual arquivo vai em qual pasta:

| Pasta | O que colocar |
|---|---|
| `diffusion_models/` | DiTs GGUF: Qwen-Image-2.1, Qwen-Image-Edit-2511, Wan 2.2 (high/low noise ou 5B TI2V) |
| `text_encoders/` | Qwen2.5-VL (Qwen) e UMT5-XXL (Wan) |
| `vae/` | `qwen_image_vae`, VAE do Wan |
| `loras/` | LoRAs de poucos passos (Lightning para Qwen-Edit, LightX2V para Wan) |
| `checkpoints/` | epiCRealism (SD 1.5) — mantém a auto-detecção e serve de fallback |
| `clip/`, `controlnet/`, `unet/`, `upscale_models/` | reservados; só se um workflow pedir |

---

## Montar o Dataset pela primeira vez

Tudo acontece **dentro de um notebook Kaggle** — nada baixa para a sua máquina.

1. Abra um notebook Kaggle (Internet **On**).
2. Em **Add-ons > Secrets**, crie `HF_TOKEN` (HuggingFace) e `CIVITAI_TOKEN` (Civitai).
3. Suba o `baixar_modelos_para_dataset.py` para o notebook (ou cole o conteúdo numa célula).
4. **Confirme os nomes de arquivo**: a lista `MODELOS` traz vários campos
   `"arquivo": "PLACEHOLDER_..."`. Abra a página de cada repo HF indicada no
   `obs`, copie o nome EXATO do `.gguf`/`.safetensors` e substitua o PLACEHOLDER.
   (O script PULA qualquer item que ainda esteja como PLACEHOLDER.)
5. Rode com `PUBLICAR = False`. Ele baixa para `/kaggle/working/comfyui-models/`
   e imprime os tamanhos. Confira se os arquivos certos chegaram.
6. Quando estiver tudo certo, mude para `PUBLICAR = True` e rode de novo. Ele gera
   o `dataset-metadata.json` e executa `kaggle datasets create -p <pasta> --dir-mode zip`.
7. Depois é só anexar esse Dataset ao notebook de execução (`+ Add Input`); o
   `kaggle_setup.py` detecta e conecta automaticamente.

> A API do Kaggle já vem disponível no notebook; se o `kaggle` CLI reclamar, habilite
> o acesso à API nas configurações do notebook.

---

## Adicionar modelos novos depois (ex.: Wan 2.2 Animate)

1. Edite a lista `MODELOS` em `baixar_modelos_para_dataset.py` acrescentando o novo
   item (origem `hf` ou `civitai`, `subpasta` correta, `repo_id`/`arquivo` ou `civitai_id`).
   - **Wan 2.2 Animate** é um DiT de vídeo → vai em **`diffusion_models/`**. Os
     encoders/VAE/LoRAs que ele exigir vão nas pastas correspondentes
     (`text_encoders/`, `vae/`, `loras/`). Use só variantes **não-NSFW**.
2. Rode o script de novo no notebook Kaggle (ele PULA o que já existe e baixa só o novo).
3. Publique uma **nova versão** do Dataset:
   - via CLI: `kaggle datasets version -p /kaggle/working/comfyui-models -m "add wan 2.2 animate" --dir-mode zip`
     (o comando está comentado na função `publicar()` do script);
   - ou pela **UI do Kaggle**: abra o Dataset > **New Version**.
4. Nos notebooks que usam o Dataset, atualize para a nova versão.

---

## Estratégia de GPU (resumo)

As 2× T4 do Kaggle são **16 GB cada e SEPARADAS** (`cuda:0` e `cuda:1`), não um
pool de 32 GB. A estratégia v1 é **jobs isolados** (não paralelismo):

- **Imagem/edição (Qwen)** na `cuda:0` — cabe inteiro numa T4 em GGUF Q4_K_M.
- **Vídeo (Wan 2.2)** na `cuda:1` — 480p, clipes curtos, poucos passos.
- **Fallback de memória**: se um clip de vídeo estourar os 16 GB, usar
  **DisTorch2 / ComfyUI-MultiGPU** para derramar blocos do DiT na segunda T4 e/ou RAM.

Importante: **DisTorch2/MultiGPU é gerenciamento de memória, NÃO ganho de
velocidade** — serve para *caber* modelo maior, não para rodar mais rápido. E o
repositório `pollockjj/ComfyUI-MultiGPU` está marcado para **arquivamento em
30/09/2026** (ainda funcional, mas sem suporte ativo).

---

## Workflows da comunidade para reaproveitar

Em vez de montar workflows do zero, reaproveite os prontos (secão 6 do arch doc):

- **Workflows oficiais do ComfyUI-MultiGPU (DisTorch2)**, pasta `example_workflows/`:
  - `qwen_image unet clip distorch2.json` (geração Qwen-Image);
  - `qwen_image_edit_2509 unet clip distorch2.json` (edição Qwen-Image-Edit);
  - `wan2_2 distorch2 double_unet no_cpu.json` (Wan 2.2 com os dois UNets);
  - `wan2_2 t2v lightx2v lora distorch2.json` e `wan2_2 t2i lightx2v lora distorch2.json`;
  - `ComfyUI-WanVideoWrapper wanvideo2_2 I2V A14B GGUF.json`.
- **Notebooks de referência Kaggle T4** (mesmo padrão Dataset + túnel deste projeto):
  - `chandan11248/qwen-image-21-t4` — Qwen-Image-2.1 GGUF em Kaggle T4 (ótima
    referência de setup e benchmarks).
  - `kelvinweijun/wan-2.2-animate-comfyui-kaggle` — Wan 2.2 em Kaggle T4 linkando
    modelos direto de Kaggle Datasets.

> ⚠️ Esses repositórios às vezes usam modelos "uncensored". **Troque sempre pelas
> variantes oficiais não-NSFW** no nosso caso.

---

## Múltiplos Datasets

O `kaggle_setup.py` atualizado detecta **mais de um Dataset anexado**: ele varre
`/kaggle/input`, aceita toda base que tenha subpasta de modelos e escreve um bloco
por Dataset no `extra_model_paths.yaml` (chaves únicas `kaggle_dataset_0`,
`kaggle_dataset_1`, ...). O ComfyUI mescla as raízes.

Na prática: se um Dataset bater no limite de tamanho do Kaggle, **separe os modelos
em dois ou mais Datasets** (ex.: um só de vídeo Wan, outro de imagem Qwen) e anexe
todos ao notebook. O setup conecta cada um automaticamente.

---

## Aviso de ToS / NSFW

A conta Kaggle é **NOVA** (a anterior foi banida por NSFW). Mantenha tudo dentro
dos termos:

- **NUNCA** baixe/use variantes NSFW, mesmo quando o repo as oferece (vários repos
  de Wan/Qwen publicam versões "uncensored" ao lado das limpas — use só as limpas).
- Conteúdo que o Kaggle proíbe vai para o **Modal** (produção, projeto `Videos_virais`),
  **nunca** para o Kaggle.

Esse cuidado está alinhado ao alerta do [`README.md`](README.md).

---

## MÉTODO VISUAL (sem código) — adicionar modelo/LoRA pela interface

> Passo a passo para adicionar um arquivo (ex.: LoRA) a um Dataset usando SÓ a
> interface do Kaggle, sem rodar script. Ideal para arquivos pequenos (LoRAs ~300 MB).
> Para ficar independente: o pulo do gato é (a) a URL de download do HF e (b) pôr o
> arquivo na PASTA certa (`loras/`, `diffusion_models/`, etc.).

### Regra de ouro das pastas
O ComfyUI só acha o arquivo se ele estiver na subpasta certa:
- LoRA → `loras/`
- modelo DiT (Qwen/Wan) → `diffusion_models/`
- text encoder → `text_encoders/`
- VAE → `vae/`
- checkpoint SD 1.5 → `checkpoints/`

### Passo a passo (ex.: LoRA Lightning do Qwen 2.1)

1. **Baixar do HuggingFace no PC**
   - Abra o repo, ex.: https://huggingface.co/NidAll/pruna-image-2.1-comfyui-loras
   - Aba **Files and versions**
   - Clique no ícone de download (↓) do arquivo
     (`p_qwen_image_2.1_8step_v0.1_comfyui.safetensors`, ~340 MB)

2. **Organizar numa pasta com o nome da subpasta**
   - No PC, crie uma pasta chamada EXATAMENTE `loras` (minúsculo)
   - Mova o `.safetensors` para dentro dela → `loras/arquivo.safetensors`

3. **Criar um Dataset novo só de LoRAs (recomendado)**
   - Kaggle → **Create → New Dataset** (ou kaggle.com/datasets → New Dataset)
   - **Arraste a PASTA `loras` inteira** (não só o arquivo) para a janela de upload
     — isso preserva a estrutura `loras/arquivo.safetensors`
   - Título: `comfyui-qwen-loras` → **Create**
   - *Por que Dataset separado:* evita re-subir os ~15 GB do Dataset grande; o
     `kaggle_setup.py` detecta múltiplos Datasets e junta tudo.

4. **Usar no notebook**
   - Painel direito → **+ Add Input** → adicione `comfyui-qwen-loras`
   - (mantenha também o `comfyui-qwen-image` anexado)
   - Rode o `kaggle_setup.py`; a LoRA aparece no nó **Load LoRA** do ComfyUI.

### Alternativa: adicionar ao Dataset existente (New Version)
- Abra o Dataset → **New Version** → arraste a pasta `loras` → publique.
- *Cuidado:* dependendo do caso, pode exigir re-subir o Dataset todo (lento se são
  15 GB). Para arquivos pequenos, o Dataset separado (passo 3) costuma ser melhor.

### Como pegar a URL de qualquer modelo do HuggingFace
Na aba **Files** do repo → 3 pontinhos ao lado do arquivo → **Copy download link**.
O padrão é sempre: `https://huggingface.co/<repo>/resolve/main/<caminho_do_arquivo>`.

---

## ATUALIZAÇÃO IMPORTANTE (estado real do projeto — out/2026)

O que foi validado na prática difere do plano v1 em pontos-chave:

- **Qwen roda em INT8 ConvRot NATIVO (não GGUF).** O DiT é
  `qwen_image_2.1_int8_convrot.safetensors`, carregado pelo nó **"Load Diffusion
  Model"** (NÃO o Unet Loader GGUF). Encoder `qwen3vl_8b_int8_convrot` (CLIPLoader,
  type `qwen_image`). VAE `qwen_image_2.1_vae_bf16`. Misturar GGUF + encoder INT8
  dá erro `input_act`.
- **Qwen-Image-2.1 já faz EDIÇÃO nativamente** (troca de roupa, inpainting,
  multi-referência até 10-16 imagens) via nó `TextEncodeQwenImage21`. NÃO precisa
  do Qwen-Image-Edit-2511 separado.
- **A T4 é lenta** (~285s/imagem a 25 steps). A única alavanca real de velocidade é
  a **LoRA Lightning** (8 steps → ~3× mais rápido). Trocar quant (Q5/Q6/Q8/BF16) NÃO
  acelera; BF16 nem cabe na T4. Encoder na CPU libera VRAM mas não acelera.
- **LoRA Lightning certa para o 2.1:** `NidAll/pruna-image-2.1-comfyui-loras`
  (arquivos `p_qwen_image_2.1_8step...` ou `5step...`). A LoRA do `lightx2v` é para
  o Qwen-Image ORIGINAL (20B), NÃO serve no 2.1. Casar steps do KSampler com a LoRA
  (8step → 8 steps; cfg 1).
- **Qwen-Image 3.0 é fechado/pago (só API)** — não roda no Kaggle. O 2.1 é o modelo
  aberto mais novo que roda local.
