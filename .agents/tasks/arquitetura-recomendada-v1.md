# Arquitetura Recomendada v1 — ComfyUI unificado no Kaggle (2× T4)

> Documento de ANÁLISE E PROPOSTA. Nada aqui foi implementado. É para você revisar
> antes de qualquer código. Escrito em pt-BR. As fontes estão linkadas ao longo do texto.
> O conteúdo das fontes foi reescrito/resumido por mim para fins de conformidade de licença.

---

## 0. Resumo executivo (a resposta curta)

- **Dá para montar um ambiente unificado (imagem + edição + vídeo) no Kaggle grátis com 2× T4? Sim**, usando **modelos quantizados GGUF** e o custom node **ComfyUI-GGUF** (city96) + **ComfyUI-MultiGPU/DisTorch2** (pollockjj). Mas com ressalvas importantes de velocidade.
- **A T4 é o gargalo, não a VRAM.** A T4 é arquitetura Turing (`sm_75`), antiga: **não tem aceleração FP8 por hardware** (FP8 real precisa Ada `sm_89+`). Então **FP8 não é o caminho na T4** — o caminho é **GGUF** (ou INT8 nativo, quando existir). Fontes da comunidade medindo isso diretamente em Kaggle 2× T4 confirmam ([chandan11248/qwen-image-21-t4](https://github.com/chandan11248/qwen-image-21-t4)).
- **MultiGPU nas T4 NÃO é paralelismo.** O ComfyUI-MultiGPU/DisTorch2 **não roda duas GPUs em paralelo para acelerar** — ele é **gerenciamento de memória**: distribui as camadas (pesos estáticos) do modelo entre `cuda:0`, `cuda:1` e a RAM do sistema, deixando mais espaço livre na GPU de cálculo. Os passos continuam **sequenciais** ([README oficial](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md)). Ou seja: as 2 T4 ajudam a **caber** modelos maiores, não a **ir mais rápido**.
- **MiniMax:** **recomendo DESCARTAR** o MiniMax (e qualquer API paga) no v1. Qwen (imagem/edição) e Wan 2.2 (vídeo) **cabem** nas T4 em GGUF, mesmo que lentos. O usuário disse: só considerar MiniMax se nada open-source couber. Como cabe, a decisão é **100% local/open-source**. (Detalhe na seção 7.)
- **A versão "2511" que o ChatGPT citou EXISTE.** `Qwen-Image-Edit-2511` é real e é a evolução da `2509`. O ChatGPT acertou. (Detalhe na seção 3.)
- **Reaproveitar workflows prontos: sim, existem.** Há workflows oficiais do ComfyUI-MultiGPU para Qwen-Image, Qwen-Image-Edit-2509 e Wan 2.2, e notebooks de comunidade rodando em Kaggle T4. (Seção 6.)

> ⚠️ **Alerta de manutenção:** o repositório `pollockjj/ComfyUI-MultiGPU` **será arquivado em 30/09/2026** e já não recebe mais updates ([aviso no README](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md)). Ele continua funcional e instalável, mas é bom saber que não há suporte ativo e que pode haver um fork futuro. Isso pesa na decisão de depender dele.

---

## 1. Realidade do hardware Kaggle (o que limita tudo)

| Recurso | Valor | Observação |
|---|---|---|
| GPUs | 2× NVIDIA Tesla T4 | **16 GB VRAM cada, SEPARADAS** (`cuda:0` e `cuda:1`). **NÃO** são um pool de 32 GB. |
| Arquitetura | Turing `sm_75` | **Sem FP8 por hardware.** FP8 "funciona" só por emulação → lento ou sem ganho. |
| RAM do sistema | ~29–32 GB | Dá para hospedar text encoder/pesos offloadados aqui. |
| Disco gravável | `/kaggle/working` ~20 GB | **Apagado ao desligar a sessão.** Não cabe guardar modelos grandes aqui. |
| Disco persistente | `/kaggle/input/<dataset>` | **Read-only, fora dos 20 GB, persiste entre sessões.** É onde os modelos ficam (via Kaggle Dataset). |
| Cota GPU | ~30 h/semana | Vídeo consome rápido; planejar os testes. |

Fonte do limite `sm_75`/FP8: o shootout técnico medido em Kaggle 2× T4 do repo [chandan11248/qwen-image-21-t4](https://github.com/chandan11248/qwen-image-21-t4) ("FP8 weights ❌ needs sm_89+ (Ada); T4 is sm_75" — barreira de hardware). Conteúdo reescrito para conformidade.

**Consequência prática nº 1:** priorizar **GGUF** sobre FP8 em toda a stack.
**Consequência prática nº 2:** usar as 2 T4 para **caber** (offload de camadas via DisTorch2), não esperar velocidade de 2 GPUs.

---

## 2. Como o ComfyUI-MultiGPU / DisTorch2 realmente funciona (e seus limites)

### O que é
"DisTorch" = "distributed torch". Ele tira as partes **estáticas** do modelo (o UNet/DiT) da sua placa de cálculo principal e as coloca em um ou mais "doadores": a RAM do sistema (CPU) **ou outra placa** (`cuda:1`). Você escolhe quanto quer liberar; o nó cuida do resto. Fonte: [README oficial](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md) (reescrito).

### O ponto mais importante (não é paralelismo)
O próprio README é explícito: **é gerenciamento de memória, não processamento paralelo.** Os passos do workflow continuam executando **sequencialmente**, só que com componentes carregados em dispositivos diferentes. O ganho é de **capacidade** (caber modelo maior / mais espaço de latente), não de **velocidade** por somar duas GPUs. ([README](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md), reescrito.)

### Modos de alocação
- **Normal:** um slider `virtual_vram_gb`. Quanto mais "VRAM virtual" você adiciona, mais do modelo é empurrado para o dispositivo doador. Simples.
- **Expert (bytes) — recomendado para 2 T4:** string tipo
  `cuda:0,8gb;cuda:1,5gb;cpu,*` → carrega 8 GB do modelo na `cuda:0`, 5 GB na `cuda:1`, e o resto na CPU/RAM.
- **Expert (ratio):** estilo `tensor_split` do llama.cpp, ex. `cuda:0,40%;cuda:1,40%;cpu,20%`.
- **Expert (fraction):** fração da VRAM total de cada dispositivo.

Exemplos verbatim do README (≤30 palavras): `cuda:0,500mb;cuda:1,3.0g;cpu,5gb*` coloca 0,5 GB em `cuda:0`, 3 GB em `cuda:1`, e o restante na CPU. Isso prova que **dá para dividir pesos entre as duas T4 + RAM** em uma única string.

### Nós relevantes para a nossa stack
- `UnetLoaderGGUFDisTorch2MultiGPU` (carrega o DiT em GGUF distribuído) — requer **ComfyUI-GGUF** (city96).
- `CLIPLoaderGGUFDisTorch2MultiGPU` / `DualCLIP.../TripleCLIP...` (text encoder distribuído).
- `VAELoaderDisTorch2MultiGPU`.
- Família **WanVideoWrapper** com nós MultiGPU dedicados (`WanVideoModelLoaderMultiGPU`, `WanVideoBlockSwapMultiGPU`, etc.) para vídeo.
Lista completa no [README](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md).

### Limitações / overhead (seja realista)
- **Latência de transferência:** mover camadas entre dispositivos (ou buscar da RAM a cada passo) adiciona overhead. Offload para RAM é mais lento que manter na VRAM; offload para a segunda GPU costuma ser melhor que para a CPU, mas ainda assim o cálculo é sequencial.
- **Não acelera:** se o modelo já cabe numa T4, DisTorch2 **não** vai deixar mais rápido — é para quando **não cabe**.
- **Manutenção encerrando (30/09/2026):** sem suporte ativo; depender dele é um risco de médio prazo.
- **DisTorch2 é "até ~10% mais rápido que DisTorch1" só para GGUF** — é um ganho sobre a versão antiga do próprio node, não sobre rodar nativo.

### Como pensar a divisão `cuda:0` / `cuda:1` conceitualmente (v1)
- **Estratégia A (recomendada p/ começar — mais simples e previsível):** rodar **uma tarefa por GPU**. Imagem/edição Qwen na `cuda:0`; vídeo Wan na `cuda:1`. Cada modelo quantizado tenta caber **inteiro em uma T4** (16 GB). Zero overhead de split. Isso é "MultiGPU" no sentido de **isolar jobs**, não de dividir um modelo. Use os nós `...MultiGPU` (sem DisTorch2) só para fixar o `device`.
- **Estratégia B (quando um modelo não cabe em 16 GB):** usar **DisTorch2 expert bytes** para derramar o excedente do DiT na segunda T4 e/ou RAM, ex. `cuda:0,13gb;cuda:1,6gb;cpu,*`. Aceita-se o overhead em troca de caber.

---

## 3. Qwen-Image e Qwen-Image-Edit — o que existe de verdade

### 3a. Esclarecendo os nomes (há duas famílias — cuidado)
- **Qwen-Image (original):** DiT grande (~20B) + text encoder Qwen2.5-VL, lançado em 2025, licença **Apache 2.0**. Base dos GGUF do [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf).
- **Qwen-Image-2.1:** modelo **7B unificado** (gera **e** edita em um só DiT), lançado em set/2026, com saída RGBA nativa e até 10 imagens de referência, licença **Qwen Research License** (não Apache) — ver [wildminder/awesome-qwen-image](https://github.com/wildminder/awesome-qwen-image) e [codersera](https://codersera.com/blog/how-to-run-qwen-image-2-1-locally-2026/). Como é **7B**, é **bem mais leve** que o 20B original e muito amigável à T4.

> **Recomendação de nomenclatura:** decidir qual "Qwen" queremos. Para **estudo + caber fácil na T4**, o **Qwen-Image-2.1 (7B)** é o mais prático (unifica geração e edição, GGUF pequenos). Para fidelidade máxima com licença Apache, o Qwen-Image 20B em GGUF Q4 também roda, porém mais pesado. **Atenção à licença:** a 2.1 é Qwen Research License (uso ok para estudo; revisar termos antes de uso comercial).

### 3b. Qwen-Image-Edit: a versão "2511" é REAL
O ChatGPT **acertou**: `Qwen-Image-Edit-2511` existe e é a evolução da `2509` (set/2025), com "melhor consistência" segundo a doc oficial do ComfyUI ([docs.comfy.org — Qwen-Image-Edit-2511](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)). Evidências adicionais:
- Template nativo embutido no ComfyUI: `image_qwen_image_edit_2511` ([localaimaster](https://localaimaster.com/blog/qwen-image-edit-local-guide)).
- GGUF e quantizações da comunidade já publicadas (ex. [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF), [calcuis/qwen-image-edit-plus-gguf](https://huggingface.co/calcuis/qwen-image-edit-plus-gguf)).

**Verdito de versão:** usar **Qwen-Image-Edit-2511** (mais recente/consistente). Guardar a **2509** como alternativa — há relatos de que a 2509 reproduz melhor ângulos/perspectiva em certos casos ([dx8152](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)).

### 3c. VRAM / quant / velocidade (imagem e edição)

| Modelo | Quant | Tamanho arquivo | VRAM aprox. | Cabe em 1 T4 (16 GB)? |
|---|---|---|---|---|
| Qwen-Image-2.1 (7B) | Q8_0 | ~7,6 GB | ~12 GB+ | ✅ sim (sobra p/ latente) ([Abiray](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF)) |
| Qwen-Image-2.1 (7B) | Q4_K_M / Q5_K_M | ~4,5–5,5 GB | ~8–10 GB | ✅ sim, folgado |
| Qwen-Image 20B | Q4_K_M | maior | aperta | ✅ provável com `--lowvram`/offload |
| Qwen-Image-Edit-2511 | Q4_K_M | ~13,2 GB | cabe em 16 GB | ✅ (recomendado p/ 16 GB) ([localaimaster](https://localaimaster.com/blog/qwen-image-edit-local-guide)) |
| Qwen-Image-Edit-2511 | Q4_0 | ~11,9 GB | p/ 12 GB | ✅ |

**Velocidade medida em Kaggle 2× T4** (repo chandan11248, Qwen-Image-2.1 7B, modelo já carregado) — reescrito:

| Tamanho / passos | GGUF Q4_K_M | INT8 oficial |
|---|---|---|
| 768² / 20 passos | ~250 s | ~115 s |
| 1024² / 25 passos | ~560 s | ~280 s |
| 512² / 10 passos | ~70 s | ~30 s |

Observações cruciais desse benchmark (reescrito de [chandan11248/qwen-image-21-t4](https://github.com/chandan11248/qwen-image-21-t4)):
- **INT8 nativo foi ~2,2× mais rápido que GGUF** na T4 (kernels int8 batem o dequant por passo do GGUF). Se existir build INT8 do modelo, **prefira INT8**; senão, GGUF Q4_K_M.
- **Cold start domina:** a 1ª imagem paga 3–4 min de carga de pesos (sobretudo o text encoder ~8–9 GB). **Sessão persistente amortiza** isso. → Mantenha o ComfyUI vivo e gere em lote.
- O **text encoder** pode ficar na **RAM/CPU** (encoding roda 1× por prompt, não pesa na velocidade de sampling) — ótimo para liberar VRAM.
- **Evitar Q8_0** em alguns Qwen GGUF: há relato de erro de shape `[136] vs [128]` no sampling; usar **Q4_K_M / Q5_K_M / Q6_K** ([Janchan123](https://huggingface.co/Janchan123/Qwen-Image-2.1-Uncensored-GGUF)). (Esse repo específico é "uncensored" — **não usar no Kaggle** por política NSFW; cito só pela nota técnica do shape.)
- **FP8 na T4: não.** Precisa `sm_89+`. **SageAttention**: upstream não dá suporte em `sm_75`. **torch.compile**: warmup longo demais para sessões curtas. (Tudo do shootout, reescrito.)
- **Lightning/distill LoRA de 4–8 passos** corta drasticamente o tempo (ex.: LoRA Lightning de ~850 MB levando 40→4 passos no Edit-2511, [localaimaster](https://localaimaster.com/blog/qwen-image-edit-local-guide)). **Fortemente recomendado** para a T4. (Nota: para a arquitetura 2.1 7B, o repo chandan reporta que a LoRA Lightning v1 **não** era compatível — validar por modelo.)

**Verdito imagem/edição:** **cabe e roda em UMA T4** (não precisa DisTorch2 para caber). Qwen-Image-2.1 7B em Q4_K_M/Q5 para geração e Qwen-Image-Edit-2511 Q4_K_M para edição. Velocidade: minutos por imagem em 768²–1024² (aceitável para estudo). Usar LoRA de poucos passos para acelerar.

---

## 4. Wan 2.2 (vídeo) — o que cabe na T4

Wan 2.2 (Alibaba, jul/2025) vem em 3 variantes ([comfyanonymous.github.io/wan22](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)):
- **5B (TI2V)** denso — faz t2v e i2v, **mais leve** (melhor candidato à T4).
- **14B T2V** e **14B I2V** — arquitetura **MoE de 2 experts** (high-noise + low-noise) que o ComfyUI carrega **sequencialmente**, mantendo o pico de VRAM mais baixo do que o tamanho total sugere ([wan-ai.app](https://wan-ai.app/wan-comfyui-workflow)).

### VRAM / quant / limites na T4
- Existe repo de GGUF **Wan2.2 I2V otimizado para T4** ([geceff/Wan2.2-Custom-Models-GGUF](https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF)). Recomendações dessa fonte (reescritas), para T4 15 GB:
  - **Resolução:** não passar de **480p de altura** e **720p de largura**.
  - **Frames:** manter em **81–120 frames no máximo** para estabilidade.
  - **Passos:** **4–12 no total** (com LoRA Lightning/distill de poucos passos), CFG 1,0–2,0.
  - **Quant via GUI:** faixa segura **Q4_K_M a Q8** para high-noise e low-noise.
  - ⚠️ Esse repo também publica modelos **FP8/"nsfw"** — **NÃO usar os NSFW** (política Kaggle). Usar apenas os GGUF "limpos" (high/low noise Q4_K_M).
- Modelos 14B GGUF rodam em VRAM muito baixa (há relatos de 6 GB com GGUF e offload) — a T4 16 GB é folgada **se** aceitarmos clipes curtos e baixa resolução ([TheLocalLab](https://www.patreon.com/TheLocalLab/posts/free-wan-2-2-14b-135754993)).
- **Dica de VRAM:** usar os nós **WanVideo...MultiGPU** + **BlockSwap** para derramar blocos do DiT na `cuda:1`/RAM quando necessário.

### Velocidade (expectativa realista)
A T4 é lenta para vídeo. Referência de hardware forte: 5B 720p/5s/24fps em ~9 min numa RTX 4090 ([localaimaster](https://localaimaster.com/blog/wan-video-generation-guide)). Na **T4**, esperar **vários minutos por 1 clipe curto** (ex. ~2–4 s, 480p, ~16 fps, poucos passos). Vídeo **consome cota** rápido — gerar com parcimônia.

**Verdito vídeo:** **cabe na T4** com **Wan 2.2 GGUF** (começar pelo **5B TI2V** ou **14B I2V Q4_K_M** com LoRA de poucos passos), **480p, ≤~81 frames, 4–12 passos**. Lento, mas funcional para estudo. Para split entre as 2 T4, usar WanVideoWrapper + nós MultiGPU/BlockSwap se estourar os 16 GB.

---

## 5. Veredito sobre DisTorch2 / MultiGPU: é necessário?

- **Para imagem/edição (Qwen): NÃO é necessário para caber** — os GGUF cabem em 1 T4. DisTorch2 só entra se você quiser rodar um modelo maior (ex. Qwen 20B Q8) ou liberar VRAM para resoluções altas.
- **Para vídeo (Wan 2.2): opcional, útil nos casos de borda** — ajuda quando o clipe/resolução estoura os 16 GB; aí você derrama blocos na `cuda:1`/RAM.
- **MultiGPU "simples" (sem DisTorch2): recomendado desde já** apenas para **fixar `device`** e rodar **imagem numa T4 e vídeo na outra** (jobs isolados). Isso é o uso mais garantido e sem overhead das 2 T4.
- **Lembre:** nada disso dá paralelismo/velocidade dobrada — é só memória. E o node **será arquivado em 30/09/2026**.

**Config conceitual recomendada v1 (expert bytes), só quando precisar derramar:**
`UnetLoaderGGUFDisTorch2MultiGPU` com `compute_device=cuda:0` e, p.ex., `expert_mode_allocations = "cuda:0,13gb;cuda:1,6gb;cpu,*"` para empurrar o excedente do DiT para a segunda T4 e depois RAM.

---

## 6. Workflows / notebooks prontos para REAPROVEITAR (não construir do zero)

O usuário quer **reusar** assets da comunidade que (a) já lidam com pouca VRAM/T4 e (b) usam quantização. Encontrados:

**Workflows oficiais do ComfyUI-MultiGPU (DisTorch2)** — JSONs prontos no repo ([README](https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/README.md), pasta `example_workflows/`):
- `qwen_image unet clip distorch2.json` — Qwen-Image geração.
- `qwen_image_edit_2509 unet clip distorch2.json` — Qwen-Image-Edit (edição).
- `wan2_2 distorch2 double_unet no_cpu.json` — Wan 2.2 com os dois UNets (high/low) distribuídos.
- `wan2_2 t2v lightx2v lora distorch2.json` e `wan2_2 t2i lightx2v lora distorch2.json` — Wan 2.2 com LoRA LightX2V (poucos passos).
- `ComfyUI-WanVideoWrapper wanvideo2_2 I2V A14B GGUF.json` — Wan 2.2 I2V 14B em GGUF.

**Notebooks Kaggle T4 prontos (padrão idêntico ao nosso: Dataset + tunnel):**
- [chandan11248/qwen-image-21-t4](https://github.com/chandan11248/qwen-image-21-t4) — Qwen-Image-2.1 GGUF em Kaggle T4, ComfyUI + ComfyUI-GGUF + tunnel + workflow JSON (`comfy/qwen21_gguf_t4_workflow.json`). **Ótima referência de setup e benchmarks.** (⚠️ o modelo que ele usa é "uncensored" — **trocar pelo Qwen-Image-2.1 oficial não-NSFW** no nosso caso.)
- [kelvinweijun/wan-2.2-animate-comfyui-kaggle](https://github.com/kelvinweijun/wan-2.2-animate-comfyui-kaggle) — Wan 2.2 em Kaggle T4 com ComfyUI, **linkando modelos (GGUF, text encoders, VAEs, LoRAs) direto de Kaggle Datasets** — exatamente o padrão `extra_model_paths`/Dataset deste projeto.

**Modelos quantizados (fontes para o Dataset):**
- Imagem: [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf), [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) (INT8/fp8 repackaged), [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF).
- Edição: [QuantStack/Qwen-Image-Edit-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF), [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI), [calcuis/qwen-image-edit-plus-gguf](https://huggingface.co/calcuis/qwen-image-edit-plus-gguf).
- Vídeo: [geceff/Wan2.2-Custom-Models-GGUF](https://huggingface.co/geceff/Wan2.2-Custom-Models-GGUF) (usar só os GGUF limpos), [calcuis/wan-gguf](https://huggingface.co/calcuis/wan-gguf), templates nativos Wan 2.2 do ComfyUI.

**Custom nodes necessários:**
- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — carregar GGUF.
- [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) — device/DisTorch2 (ciente do arquivamento).
- [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) — vídeo Wan (integração MultiGPU dedicada).

---

## 7. Veredito sobre MiniMax (API paga)

**Decisão v1: NÃO usar MiniMax.** Justificativa objetiva, baseada nos achados de VRAM:
- O usuário foi claro: só considerar MiniMax **se não houver** modelo open-source que caiba na T4.
- **Qwen-Image-2.1 (imagem) cabe** em 1 T4 (GGUF). ✅
- **Qwen-Image-Edit-2511 (edição) cabe** em 1 T4 (Q4_K_M ~13 GB). ✅
- **Wan 2.2 (vídeo) cabe** em 1 T4 (GGUF, 480p, clipe curto, poucos passos). ✅

Como **os três cabem**, não há "lacuna" que justifique API paga. Vai **tudo local/open-source**. O único "custo" é **velocidade/tempo** (T4 é lenta), não "não roda". Se, na validação, algum cenário específico se mostrar **inviável** na prática (ex. vídeo longo/alta resolução que a cota de 30h/semana não comporta), aí sim reabrir a conversa sobre uma API — mas isso seria exceção pontual, não arquitetura. Lembrando que o projeto já reserva **produção de vídeo "pesada" para o Modal/`Videos_virais`**, não para o Kaggle.

---

## 8. Estrutura do Kaggle Dataset (compatível com o `kaggle_setup.py` atual)

O `kaggle_setup.py` (lido em `c:\Users\Taty\Desktop\Comfyui_kaggle\kaggle_setup.py`) detecta automaticamente a pasta que **contém `checkpoints`** dentro de `/kaggle/input` (`detectar_dataset_base`) e escreve um `extra_model_paths.yaml` com estas chaves: `checkpoints`, `loras`, `vae`, `clip`, `unet`, `controlnet`, `upscale_models`.

**Problema:** os modelos GGUF de **Qwen DiT** e **Wan DiT** no ComfyUI nativo ficam em **`diffusion_models/`**, e os **text encoders** em **`text_encoders/`** — e o YAML atual **não mapeia** essas duas pastas. O ComfyUI-GGUF (`UnetLoaderGGUF`) lê de `models/unet` **ou** `models/diffusion_models`; os guias oficiais mandam o DiT para **`diffusion_models/`** e o encoder para **`text_encoders/`** ([comfyanonymous wan22](https://comfyanonymous.github.io/ComfyUI_examples/wan22/), [city96/Qwen-Image-gguf](https://huggingface.co/city96/Qwen-Image-gguf), [HF discuss Wan 5B](https://discuss.huggingface.co/t/a-model-for-converting-photos-to-videos-with-8gb-of-vram/170034/6)).

**Layout de Dataset proposto (v1):**
```
comfyui-models/                 (raiz do Dataset; precisa conter 'checkpoints' p/ auto-detecção)
├── checkpoints/                # mantém o auto-detect feliz (pode pôr o epiCRealism aqui)
│   └── epicrealism_naturalSinRC1VAE.safetensors
├── diffusion_models/           # DiTs em GGUF (Qwen-Image, Qwen-Image-Edit, Wan 2.2)
│   ├── qwen-image-2.1-Q4_K_M.gguf
│   ├── qwen-image-edit-2511-Q4_K_M.gguf
│   ├── wan2.2_i2v_high_noise_14B_Q4_K_M.gguf
│   └── wan2.2_i2v_low_noise_14B_Q4_K_M.gguf
├── text_encoders/              # Qwen2.5-VL (imagem/edição) e UMT5 (Wan)
│   ├── qwen_2.5_vl_7b.safetensors         # ou .gguf
│   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
├── vae/
│   ├── qwen_image_vae.safetensors
│   └── wan_2.1_vae.safetensors            # (e wan2.2_vae p/ o 5B, se usar)
├── loras/                      # LoRAs Lightning/LightX2V de poucos passos
│   ├── qwen-image-edit-2511-lightning-4steps.safetensors
│   └── wan2.2_lightx2v_4steps_rank64.safetensors
├── clip/                       # (reservado; CLIP padrão se algum workflow pedir)
├── controlnet/
├── unet/                       # (alternativa a diffusion_models p/ ComfyUI-GGUF)
└── upscale_models/
```

**Ajuste necessário no `kaggle_setup.py` (anotar, NÃO implementar agora):** acrescentar duas linhas ao YAML gerado em `configurar_dataset()`:
```yaml
    diffusion_models: diffusion_models
    text_encoders: text_encoders
```
Sem isso, os GGUF de Qwen/Wan **não aparecem** nos nós `UnetLoaderGGUF` / `CLIPLoader` lendo do Dataset. (É a única mudança estrutural que a arquitetura exige no script atual.)

Também será preciso adicionar à lista `CUSTOM_NODES` do script: `ComfyUI-GGUF`, `ComfyUI-MultiGPU` e `ComfyUI-WanVideoWrapper` (hoje só tem o Inspire Pack).

> **Nota de disco:** o Dataset é read-only e persiste; os GGUF (vários GB cada) **nunca** vão para `/kaggle/working`. Some os tamanhos antes de montar o Dataset — Qwen-Edit Q4 (~13 GB) + Wan high/low (~2× alguns GB) + encoders (UMT5 fp16 ~11,4 GB / Qwen2.5-VL) somam bastante; priorizar quants menores e encoders fp8/gguf para o Dataset não ficar gigante.

---

## 9. Lógica de pipeline recomendada (local-first)

1. **Imagem (t2i / i2i):** Qwen-Image-2.1 7B GGUF (Q4_K_M/Q5) na `cuda:0`. Text encoder na RAM. LoRA de poucos passos se compatível. 768²–1024².
2. **Edição:** Qwen-Image-Edit-2511 Q4_K_M na `cuda:0` + LoRA Lightning 4 passos.
3. **Vídeo:** Wan 2.2 GGUF (5B TI2V ou 14B I2V Q4_K_M) na `cuda:1`, 480p, ≤~81 frames, 4–12 passos, LoRA LightX2V. BlockSwap/MultiGPU se estourar VRAM.
4. **Sessão persistente:** subir o ComfyUI uma vez e gerar em lote (amortiza o cold start de 3–4 min).
5. **MiniMax/API:** fora do v1 (ver seção 7). Produção de vídeo pesada → Modal (`Videos_virais`), não Kaggle.
6. **ToS/NSFW:** usar **apenas** modelos "limpos" (não baixar variantes NSFW mesmo quando o repo as oferece). A conta nova depende disso.

---

## 10. Próximos passos (decidir/validar antes de implementar)

1. **Escolher a família Qwen:** Qwen-Image-2.1 (7B, leve, Qwen Research License) **vs** Qwen-Image 20B (Apache, mais pesado). Recomendo **2.1** para estudo. — *decisão sua.*
2. **Confirmar licença** da Qwen Research License para o uso pretendido (estudo ok; checar se há intenção comercial).
3. **Validar na prática 1 modelo por vez** numa sessão Kaggle 2× T4: (a) Qwen-Image-2.1 Q4_K_M gerando 768²; (b) Qwen-Image-Edit-2511 Q4_K_M; (c) Wan 2.2 GGUF 480p clipe curto. Medir VRAM real (`nvidia-smi`) e tempo.
4. **Testar se precisa DisTorch2** ou se cabe em 1 T4 sem split (provável que caiba p/ imagem/edição).
5. **Decidir a estratégia de GPU:** jobs isolados (imagem na `cuda:0`, vídeo na `cuda:1`) vs split de modelo com DisTorch2.
6. **Avaliar o risco do arquivamento** do ComfyUI-MultiGPU (30/09/2026) — se vamos depender dele ou preferir manter tudo cabendo em 1 T4 sem ele.
7. **Montar o Kaggle Dataset** com o layout da seção 8 e **ajustar o `kaggle_setup.py`** (2 linhas no YAML + custom nodes). — só depois da validação.
8. **Confirmar LoRAs de poucos passos** compatíveis com cada arquitetura exata escolhida (há incompatibilidades por versão).

---

### Fontes principais (todas linkadas inline acima)
- ComfyUI-MultiGPU / DisTorch2 — README oficial (pollockjj) — aviso de arquivamento 30/09/2026.
- chandan11248/qwen-image-21-t4 — benchmarks reais Kaggle 2× T4, barreira FP8/`sm_75`.
- geceff/Wan2.2-Custom-Models-GGUF — limites de resolução/frames/passos na T4.
- docs.comfy.org, localaimaster, codersera, wildminder/awesome-qwen-image — versões e VRAM Qwen.
- comfyanonymous.github.io/wan22, HF discuss, city96/Qwen-Image-gguf, QuantStack — pastas e colocação dos arquivos.
- kelvinweijun/wan-2.2-animate-comfyui-kaggle — padrão Kaggle Dataset + ComfyUI (igual ao nosso).

> Conteúdo das fontes foi reescrito/resumido por mim para conformidade com restrições de licenciamento.
