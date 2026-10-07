# PROGRESSO — ComfyUI no Kaggle (estado atual do projeto)

> Documento de ESTADO para retomar o trabalho sem perder contexto. Atualize ao
> fim de cada etapa. Última atualização: 2026-10-06.

## Objetivo do projeto

Lab de ESTUDO de ComfyUI rodando no Kaggle grátis (2× T4 16GB). Imagem (Qwen),
edição (Qwen-Edit), vídeo (Wan 2.2). Tudo local/open-source, GGUF/INT8, sem API
paga. Conta Kaggle NOVA — NUNCA usar modelos NSFW (conta anterior foi banida).

- GitHub (de onde o Kaggle puxa os scripts): **marconi2/Comfyui_kaggle**
- Usuário Kaggle (dono dos Datasets): **letroprintdigital** (NÃO é o do GitHub!)
- Secret da API Kaggle: label **`comfyui`** (contém a KAGGLE_KEY). Também HF_TOKEN.

## Arquivos do projeto

- `kaggle_setup.py` — instala ComfyUI + Manager + custom nodes + detecta Dataset(s)
  + sobe túnel cloudflared. Rodar no Kaggle com:
  `!wget -q https://raw.githubusercontent.com/marconi2/Comfyui_kaggle/main/kaggle_setup.py -O setup.py && python setup.py`
- `baixar_modelos_para_dataset.py` — baixa modelos DENTRO do Kaggle e publica como
  Kaggle Dataset (por GRUPOS, pra caber nos 20GB do working). Flag `PUBLICAR`.
- `.agents/tasks/arquitetura-recomendada-v1.md` — análise/arquitetura v1.
- `GUIA_MODELOS_DATASET.md` — guia de modelos/dataset.

## Estratégia de armazenamento (resolvida)

- Working `/kaggle/working` = ~20GB, apagado ao desligar. Monta ComfyUI aqui.
- Kaggle Dataset montado em `/kaggle/input/datasets/<user>/<slug>/` = read-only,
  persiste, fora dos 20GB. Modelos ficam aqui.
- UM DATASET POR GRUPO (cabe nos 20GB cada): `comfyui-qwen-image`, (futuro)
  `comfyui-qwen-edit`, `comfyui-wan`. Anexa só o que vai usar.
- O setup detecta MÚLTIPLOS datasets (varredura até profundidade 5) e escreve
  extra_model_paths.yaml com um bloco por dataset.

## Lições aprendidas (erros já resolvidos — NÃO repetir)

1. **Internet do notebook tem que estar On** senão wget falha (name resolution).
2. **GPU T4 x2 tem que estar ligada ANTES** de rodar o ComfyUI, senão
   "Torch not compiled with CUDA enabled" (ambiente vira CPU-only).
3. **Dataset monta sob demanda** — só baixa quando uma célula ACESSA /kaggle/input.
   Montar 14GB demora; a célula fica "girando" (normal, não travou).
4. **kaggle datasets create**: use `--dir-mode zip` (o `skip` IGNORA subpastas e
   sobe vazio). O Kaggle descompacta o zip ao anexar; estrutura de pastas fica OK.
5. **Invalid Owner Id** = usuário errado no id do Dataset. É `letroprintdigital`
   (Kaggle), não `marconi2` (GitHub).
6. **hf_hub_download com caminho interno** (ex. split_files/...) baixa aninhado;
   o script ACHATA pro nome base e limpa pastas/.cache antes de publicar.
7. **Qwen-Image-2.1 usa encoder DIFERENTE do Qwen-Image original.**
   - ERRADO: `qwen_2.5_vl_7b_fp8_scaled` (dim 3584) -> erro `normalized_shape 4096 vs 3584`.
   - CERTO: `qwen3vl_8b_int8_convrot.safetensors` (Qwen3-VL-8B, dim 4096).
8. **nvfp4** só roda em GPU Blackwell (RTX 50xx), NÃO na T4. Usar fp8/int8.
9. **CFG do Qwen 2.1 = 1** (não 2.5). Sampler euler, scheduler simple, 20-25 steps.
10. **CONFLITO GGUF <-> INT8 ConvRot (erro `input_act`)**: carregar o DiT via nó
    GGUF (Unet Loader GGUF) + encoder INT8 ConvRot = erro
    `forward_ggml_cast_weights() got an unexpected keyword argument 'input_act'`.
    Vale tanto pro city96 quanto pro leejet (ComfyUI 0.39 muito novo).
    SOLUÇÃO (decisão do usuário = opção B): usar o DiT em **INT8 ConvRot nativo**
    (`qwen_image_2.1_int8_convrot.safetensors`) carregado pelo nó NATIVO
    **"Load Diffusion Model"** (NÃO o Unet GGUF). Sem nó GGUF no grafo = sem conflito.
    O ComfyUI-GGUF fica instalado (útil pro Wan depois), só não é usado no Qwen.

## Dataset `comfyui-qwen-image` — conteúdo CERTO (grupo qwen-image)

Repo oficial de origem: **Comfy-Org/Qwen-Image-2.1** (limpo, não-NSFW).

| Subpasta | Arquivo | Tamanho | Carregar com |
|---|---|---|---|
| diffusion_models | `qwen_image_2.1_int8_convrot.safetensors` | ~7,25 GB | **Load Diffusion Model** (nativo) |
| text_encoders | `qwen3vl_8b_int8_convrot.safetensors` | ~9,35 GB | **CLIPLoader**, type `qwen_image` |
| vae | `qwen_image_2.1_vae_bf16.safetensors` | ~0,68 GB | **Load VAE** |

(O `qwen_image_2.1_Q6_K.gguf` do início foi DESCARTADO — era GGUF, dava conflito.)

## Workflow do Qwen 2.1 (txt2img) — nós e ligações

1. **Load Diffusion Model** (nativo) -> `qwen_image_2.1_int8_convrot.safetensors`
2. **CLIPLoader** -> `qwen3vl_8b_int8_convrot.safetensors`, type `qwen_image`
3. **Load VAE** -> `qwen_image_2.1_vae_bf16.safetensors`
4. **CLIP Text Encode** (positivo) + **CLIP Text Encode** (negativo, vazio)
5. **Empty Latent Image** -> 1024x1024 (ou 768 pra ir rápido)
6. **KSampler** -> steps 20-25, **cfg 1**, euler, simple, denoise 1
7. **VAE Decode** -> **Save Image**
Ligações: Model->KSampler; CLIP->os 2 encode; encode+/- ->KSampler pos/neg;
Latent->KSampler; KSampler->VAEDecode(samples); VAE->VAEDecode(vae); ->SaveImage.

## O que JÁ FUNCIONA

- Repo GitHub criado e setup puxando dele. OK
- Dataset montando certo em /kaggle/input (estrutura de pastas intacta). OK
- ComfyUI sobe, detecta Dataset, 2× T4 ativas, túnel cloudflared abre. OK
  (se der erro 1033 no cloudflare: esperar ~1 min e dar F5 — é o servidor subindo)
- Custom nodes instalados: Inspire, ComfyUI-GGUF (leejet), ComfyUI-MultiGPU,
  ComfyUI-WanVideoWrapper. OK
- Encoder/VAE do Qwen 2.1 carregam sem erro de dimensão. OK
- **QWEN-IMAGE-2.1 GERANDO IMAGEM NA T4! (2026-10-06)** — DiT INT8 ConvRot via
  "Load Diffusion Model" nativo + encoder qwen3vl int8 + VAE 2.1, cfg 1. FASE 1 OK.
- epiCRealism (SD 1.5): baixado automático pelo setup pra checkpoints/ do working
  SE o Secret CIVITAI_TOKEN estiver anexado. Usa nó "Load Checkpoint" comum.

## PRÓXIMO PASSO (onde paramos)

FASE 1 (Qwen txt2img) CONCLUÍDA. ✅ Próximas fases:
1. (opcional) Explorar o Qwen: resoluções, prompts, LoRAs de poucos passos.
2. Grupo `comfyui-qwen-edit` (Qwen-Image-Edit-2511) — edição de imagem.
3. Grupo `comfyui-wan` (Wan 2.2 vídeo) — aí SIM usa nós GGUF/MultiGPU.

## DEPOIS (fases seguintes)

- Grupo `comfyui-qwen-edit` (Qwen-Image-Edit-2511) + LoRA Lightning.
- Grupo `comfyui-wan` (Wan 2.2 GGUF 480p) — aí SIM usa os nós GGUF/MultiGPU.
- Ver estratégia multi-GPU (jobs isolados) e fallback DisTorch2 pro Wan.

## Dicas operacionais Kaggle

- Célula do servidor ComfyUI PRENDE o kernel (fica viva servindo). Pra rodar
  outra coisa, pare essa célula (stop) antes.
- Pra editar config do script sem re-baixar modelos: importa como módulo e
  seta atributos (`import ... as b; b.PUBLICAR=True; b.main()`).
- `git push` do lado do PC -> Kaggle re-roda a mesma célula wget e pega a versão
  nova. Nunca copia/cola código.
