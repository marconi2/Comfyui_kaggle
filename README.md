# Comfyui_kaggle — Laboratório de ESTUDO de ComfyUI no Kaggle

Projeto SEPARADO do projeto de produção (`Videos_virais`). Aqui é só **estudo/
aprendizado** de ComfyUI gerando IMAGENS na GPU grátis do Kaggle. A produção de
vídeo (Wan/Modal) fica no OUTRO projeto (`Videos_virais`) e NÃO entra aqui.

> Divisão combinada pelo usuário:
> - **Modal** = produção (vídeo Wan). Projeto `Videos_virais`.
> - **Kaggle** = estudo (imagem/ComfyUI). ESTE projeto.

---

## CONTEXTO PARA O ASSISTENTE (ler primeiro)

Este projeto foi separado de `Videos_virais` em 2026-10-05. O que foi feito:
- A pasta era uma CÓPIA completa do `Videos_virais` e apontava para o MESMO repo
  GitHub (`marconi2/Videos_virais`) — risco de sobrescrever a produção.
- CORRIGIDO: removido o `.git` antigo (não aponta mais para a produção).
- LIMPO: removidos os arquivos de produção de vídeo (`modal_app/`, `scripts/`,
  `custom_nodes/`, `ComfyUI_Modal.bat`, `audio_generator.py`, `entrada/`, `saida/`,
  `workflows/`). Eles seguem seguros no projeto `Videos_virais` original.
- Sobrou só o essencial de estudo Kaggle (ver "Arquivos" abaixo).

### PENDÊNCIAS (o que falta fazer — fazer na nova sessão do VS Code)

1. **Criar um repositório GitHub NOVO e SEPARADO** para este projeto
   (sugestão de nome: `Comfyui_kaggle`). NÃO reusar `Videos_virais`.
2. **`git init`** nesta pasta + conectar ao repo novo + primeiro commit/push.
3. **Atualizar a URL dentro de `kaggle_setup.py`**: hoje aponta para
   `raw.githubusercontent.com/marconi2/Videos_virais/main/kaggle_setup.py`.
   Trocar para o repo NOVO (ex.: `marconi2/Comfyui_kaggle/main/kaggle_setup.py`).
   Essa URL é a que o Kaggle usa para baixar o setup — tem que bater com o repo novo.

### ALERTA IMPORTANTE (ban do Kaggle)

A conta Kaggle ANTERIOR foi BANIDA por conteúdo NSFW. Esta é uma conta NOVA.
- MANTER o conteúdo gerado no Kaggle DENTRO dos termos (sem NSFW), senão a conta
  nova será banida de novo (o Kaggle pode vincular contas por device/telefone/IP).
- Para conteúdo que o Kaggle proíbe, o lugar é o Modal (produção), nunca o Kaggle.

---

## O que este projeto faz

Rodar ComfyUI DENTRO de um notebook Kaggle (GPU T4 grátis), acessado pelo navegador
via túnel cloudflared, para aprender a montar workflows e gerar IMAGENS (ex.:
epiCRealism SD 1.5, Flux FP8). NÃO roda vídeo Wan 14B (T4 é pequena demais).

Arquitetura: o ComfyUI roda no Kaggle (backend + T4); você acessa a UI pelo
navegador via a URL `trycloudflare.com` que o setup imprime.

## Arquivos

```
Comfyui_kaggle/
├── README.md               # este arquivo
├── kaggle_setup.py         # SCRIPT PRINCIPAL: instala ComfyUI + Manager + nodes
│                           # + conecta Dataset de modelos + sobe servidor + túnel
├── kaggle_setup_unico.md   # versão "uma célula só" (alternativa ao GitHub)
├── kaggle_comfyui_flux.md  # guia didático passo a passo (célula por célula)
└── .gitignore
```

## Como usar no Kaggle (fluxo pretendido)

Pré-requisitos no notebook Kaggle:
- Conta Kaggle com telefone verificado (libera GPU).
- Accelerator = **GPU T4 x2**, Internet = **On**.
- **Secrets** (Add-ons > Secrets): `HF_TOKEN` (HuggingFace) e `CIVITAI_TOKEN` (Civitai).
- Dataset de modelos anexado (opcional; ver abaixo).

Rodar (UMA célula — depois de o repo novo existir e a URL estar atualizada):
```
!wget -q https://raw.githubusercontent.com/marconi2/Comfyui_kaggle/main/kaggle_setup.py -O setup.py && python setup.py
```
Isso instala tudo e imprime a URL `trycloudflare.com` para abrir no navegador.

## Estratégia de modelos (Kaggle Dataset)

O `/kaggle/working` tem só ~20 GB e é apagado ao desligar a sessão. Por isso os
modelos ficam num **Kaggle Dataset** (`comfyui-models`), montado em `/kaggle/input/`
(persistente, fora dos 20 GB, read-only). O `kaggle_setup.py` DETECTA o Dataset
automaticamente (procura a pasta `checkpoints` em `/kaggle/input`) e aponta o
ComfyUI para lá via `extra_model_paths.yaml`.

Estrutura dentro do Dataset: `checkpoints/`, `loras/`, `vae/`, `clip/`, `unet/`, etc.

## O que o kaggle_setup.py faz (resumo técnico)

1. Instala ComfyUI (git clone + requirements).
2. Instala ComfyUI-Manager.
3. Instala custom nodes fixos (lista `CUSTOM_NODES` no topo do script — hoje:
   Inspire Pack). Para adicionar node novo: inclua a URL do repo git na lista.
4. Detecta e conecta o Dataset de modelos (extra_model_paths.yaml).
5. Baixa o epiCRealism no working como fallback (se não estiver no Dataset) —
   usa o Secret `CIVITAI_TOKEN` (modelVersionId 143906 do Civitai).
6. Baixa o cloudflared e sobe o ComfyUI (porta 8188) + túnel.

Tokens: lidos dos **Secrets do Kaggle** via `UserSecretsClient` (nunca colados em
célula nem versionados no GitHub).

## Modelos de estudo

- **epiCRealism** (SD 1.5, ~2 GB) — Civitai modelVersionId 143906. Leve, roda bem na T4.
- **Flux FP8** (~17 GB) — mais pesado; cabe apertado na T4 (lento). Via Dataset.
- Workflows SD 1.5 usam epiCRealism; workflows Flux usam Flux (NÃO intercambiáveis).

## Histórico relevante (para o assistente entender a jornada)

- O setup via túnel cloudflared no Kaggle JÁ funcionou antes (ComfyUI + Manager +
  Dataset + epiCRealism), na conta anterior. O código aqui é o mesmo, validado.
- A conta anterior foi banida (NSFW), não por problema técnico do setup.
- NOTA sobre custo/risco: Kaggle é grátis mas tem política anti-NSFW e pode banir
  por túneis/uso atípico. Alternativa paga sem esse risco = Modal (projeto de produção).
