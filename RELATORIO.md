# Relatório técnico — Laboratório de inferência de LLM com Kubernetes, GPU e GuideLLM

**Data de referência:** 16/10/2026  
**Projeto local:** `~/Documentos/ml-lab-benchmarks`  
**Sistema operacional:** Ubuntu 26.04 bare metal  
**Hardware:** Intel Core i5 de 3ª geração, 16 GB DDR3, NVIDIA RTX 3050 6 GB  
**Orquestração:** Minikube 1.39.0 / Kubernetes v1.37.0  
**Inferência:** Ollama + `qwen2.5:3b` (Q4_K_M)  
**Benchmark:** GuideLLM 0.8.1 em container Docker

## 1. Objetivo do laboratório

Montar um ambiente local para aprender Kubernetes e inferência de LLMs com GPU, validar a cadeia de execução e estabelecer uma linha de base de desempenho que possa ser repetida e comparada.

O fluxo implementado foi:

1. Configurar o host para executar containers com acesso à GPU NVIDIA.
2. Criar um perfil Minikube dedicado chamado `gpu-lab`.
3. Registrar a GPU como recurso alocável no Kubernetes por meio do NVIDIA Device Plugin.
4. Executar um Pod de teste CUDA e verificar `nvidia-smi` dentro do cluster.
5. Implantar Ollama no namespace `ml-lab`, solicitar uma GPU e persistir os arquivos do modelo em PVC.
6. Validar a API do Ollama por meio de port-forward.
7. Executar GuideLLM em um container separado e salvar resultados JSON e CSV.

## 2. Hardware e software

| Componente | Estado/configuração observada |
|---|---|
| Sistema operacional | Ubuntu 26.04 bare metal |
| CPU | Intel Core i5 de 3ª geração |
| Memória RAM | 16 GB DDR3 (aprox. 15 GiB visíveis ao sistema) |
| GPU | NVIDIA RTX 3050 6 GB |
| Driver NVIDIA | 595.99.02 |
| CUDA reportado por `nvidia-smi` | 13.2 |
| Docker Engine no host | 29.9.0 |
| Minikube | v1.39.0 |
| Kubernetes no nó Minikube | v1.37.0 |
| Runtime reportado pelo nó | `docker://29.7.2` |
| GuideLLM | 0.8.1 |
| Modelo | `qwen2.5:3b`, quantização Q4_K_M, aprox. 1,93 GB |

**Observação:** as versões reportadas pelo Docker no host e pelo runtime do nó Minikube não são iguais. Isso é possível porque o Minikube executa seu próprio ambiente de nó dentro de um container.

## 3. Perfil Minikube e acesso à GPU

Foi criado um perfil separado para o laboratório, preservando o perfil antigo `minikube`:

```bash
minikube start -p gpu-lab \
  --driver=docker \
  --container-runtime=docker \
  --gpus=all \
  --cpus=3 \
  --memory=8192
```

O perfil `gpu-lab` ficou ativo e o nó `gpu-lab` reportou estado `Ready`. O perfil antigo foi parado, não excluído.

### 3.1 NVIDIA Container Toolkit e teste no host

O Docker no host foi configurado para usar a GPU NVIDIA. O teste abaixo foi concluído com sucesso:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Isso comprova o acesso à GPU a partir de um container Docker no host, mas, isoladamente, não comprova que o Kubernetes consegue alocá-la a um Pod.

### 3.2 NVIDIA Device Plugin

O NVIDIA Device Plugin foi iniciado no cluster e apareceu como um Pod DaemonSet com nome semelhante a `nvidia-device-plugin-daemonset-j6zvc`.

O plugin detecta a GPU e registra no Kubernetes o recurso estendido:

```text
nvidia.com/gpu
```

Validação feita no nó:

```bash
minikube kubectl -p gpu-lab -- describe node gpu-lab
```

O resultado mostrou:

```text
Capacity:
  nvidia.com/gpu: 1
Allocatable:
  nvidia.com/gpu: 1
```

O nome exato do Pod do Device Plugin pode mudar após reinicializações. O importante é que o recurso `nvidia.com/gpu: 1` ficou disponível para agendamento.

> **Importante:** o NVIDIA Device Plugin é um componente de infraestrutura do cluster, separado do Deployment do Ollama. O arquivo `manifest.yaml` incluído com este relatório define a solicitação de GPU pelo workload e contém um Pod opcional de teste; não substitui a instalação do Device Plugin.

### 3.3 Teste de GPU dentro do Kubernetes

Foi executado um Pod de teste com a imagem CUDA e o comando `nvidia-smi`, solicitando uma GPU:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

O Pod executou `nvidia-smi` e identificou a RTX 3050, o driver 595.99.02 e CUDA 13.2. O teste comprovou a cadeia de disponibilização da GPU ao workload Kubernetes.

## 4. Ollama no namespace `ml-lab`

### 4.1 Recursos implantados

| Recurso | Nome | Função |
|---|---|---|
| Namespace | `ml-lab` | Isola os recursos do laboratório |
| PersistentVolumeClaim | `ollama-models` | Reserva 10 GiB para os dados/modelos do Ollama |
| Deployment | `ollama` | Mantém uma réplica do servidor Ollama |
| Service | `ollama` | Expõe o serviço dentro do cluster em TCP/11434 |
| Pod | `ollama-5dd8f6bdf7-qkvm8` (observado) | Instância do servidor no momento da verificação |

O PVC ficou `Bound`. O Pod do Ollama foi observado em estado `Running`, `Ready`, sem reinicializações.

### 4.2 Decisões de configuração

- Imagem: `ollama/ollama:latest`.
- Endereço de escuta: `0.0.0.0:11434`, para aceitar conexões pela rede do Pod.
- Persistência: PVC montado em `/root/.ollama`.
- GPU: limite de uma GPU via `nvidia.com/gpu: 1`.
- CPU: request `500m`, limit `2`.
- Memória: request `3Gi`, limit `6Gi`.
- Concorrência: `OLLAMA_NUM_PARALLEL=1`.
- Modelos simultaneamente carregados: `OLLAMA_MAX_LOADED_MODELS=1`.
- Permanência em memória após uso: `OLLAMA_KEEP_ALIVE=5m`.
- Startup/readiness probes: `GET /api/tags` na porta 11434.

Os limites de CPU e memória são limites de recursos do container; o limite `nvidia.com/gpu: 1` reserva a GPU inteira para esse Pod segundo o mecanismo de alocação do Kubernetes. Não significa que o Kubernetes particione a GPU em frações.

### 4.3 Modelo

O modelo `qwen2.5:3b` foi baixado no Ollama. Os dados reportados indicavam:

- Arquitetura/modelo: Qwen 2.5 3B.
- Quantização: Q4_K_M.
- Tamanho aproximado: 1,93 GB.
- Número de parâmetros: aproximadamente 3,1 bilhões.
- Contexto declarado nos metadados: 32.768 tokens; a inicialização observada do Ollama indicou contexto efetivo de 4.096 tokens naquele momento.

O PVC permite persistir os arquivos sob `/root/.ollama` entre recriações do Pod, desde que o volume seja preservado.

## 5. Acesso à API do Ollama

A porta 11434 do host já era usada por uma instância local do Ollama. Para evitar conflito, o port-forward do Kubernetes foi feito usando a porta 11435 no host:

```bash
minikube kubectl -p gpu-lab -- \
  port-forward -n ml-lab service/ollama 11435:11434
```

Esse comando deve permanecer em execução em um terminal enquanto os testes locais usam o endpoint encaminhado.

### 5.1 Verificações executadas

Listar modelos:

```bash
curl http://localhost:11435/api/tags
```

Verificar compatibilidade com a API OpenAI:

```bash
curl http://localhost:11435/v1/models
```

Foi também feita uma requisição a `/v1/chat/completions`, com resposta bem-sucedida.

### 5.2 Evidência de inferência na GPU

Os logs do Ollama indicaram que **37 de 37 camadas foram transferidas para a GPU** (`offloaded 37/37 layers to GPU`). Também foram observados buffers do modelo e do cache KV na memória da GPU.

Em uma chamada manual, após o carregamento do modelo, foram reportados aproximadamente 593 tokens gerados em 9,91 segundos, cerca de 59,72 tokens/s segundo os logs do Ollama. Isso é uma observação pontual, não um resultado de benchmark controlado.

A primeira requisição foi muito mais lenta, com tempo total aproximado de 84,97 segundos, incluindo carregamento inicial. Essa diferença ilustra por que é importante separar aquecimento/carregamento de medições em regime estável.

## 6. Container do GuideLLM

O Python do host era a versão 3.14.4, enquanto a documentação/versão usada do GuideLLM suportava Python até 3.13. Para não alterar o Python do sistema, o GuideLLM foi isolado em um container.

### 6.1 Estrutura local

```text
~/Documentos/ml-lab-benchmarks/
├── Dockerfile
└── results/
    ├── baseline.json
    └── baseline.csv
```

### 6.2 Dockerfile utilizado

```dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir "guidellm[recommended]"
ENTRYPOINT ["guidellm"]
```

A imagem `guidellm-lab:latest` foi construída com sucesso. A imagem era grande (aproximadamente 9,87 GB de uso de disco reportado pelo Docker; tamanho de conteúdo reportado em torno de 3,32 GB), portanto não é necessário reconstruí-la para repetir o benchmark.

Verificação da CLI:

```bash
docker run --rm guidellm-lab:latest --help
```

A versão apresentada foi GuideLLM 0.8.1.

### 6.3 Por que usar `--network host`

O benchmark foi executado no Linux com `--network host` para que o container GuideLLM pudesse acessar `127.0.0.1:11435`, porta publicada pelo `kubectl port-forward` no host.

Isso pressupõe que o port-forward esteja ativo em outro terminal. Se ele não estiver, o GuideLLM não conseguirá alcançar o servidor.

### 6.4 Ajustes feitos para validar o backend

O primeiro teste falhou porque o validador padrão do GuideLLM tentou acessar `/health`, endpoint que o Ollama não disponibiliza nessa API e que respondeu 404. A validação automática foi desabilitada com `validate_backend=false`.

A segunda tentativa falhou porque o GuideLLM tentou tratar `qwen2.5:3b` como identificador de tokenizer Hugging Face. A configuração foi corrigida especificando o tokenizer `Qwen/Qwen2.5-3B-Instruct`.

O tokenizer é usado pelo GuideLLM para preparar/contabilizar tokens e é independente dos pesos do modelo carregados pelo Ollama. Não é necessário baixar novamente os pesos do modelo Ollama para fazer essa correção.

## 7. Benchmark baseline executado

### 7.1 Comando reproduzível

Execute com o port-forward ativo:

```bash
cd ~/Documentos/ml-lab-benchmarks

docker run --rm \
  --network host \
  -v "$PWD/results:/results" \
  guidellm-lab:latest run \
  --backend kind=openai_http,target=http://127.0.0.1:11435,model=qwen2.5:3b,request_format=/v1/chat/completions,validate_backend=false \
  --tokenizer kind=huggingface_auto,model=Qwen/Qwen2.5-3B-Instruct \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=5 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=128 \
  --output kind=json,path=/results/baseline.json \
  --output kind=csv,path=/results/baseline.csv
```

O comando limita a execução a cinco requisições síncronas e grava os resultados no diretório local `results/`.

### 7.2 Resultado observado

O GuideLLM informou que o benchmark foi concluído e gerou um benchmark.

| Métrica | Resultado observado |
|---|---:|
| Requisições concluídas | 5 |
| Requisições incompletas | 0 |
| Erros | 0 |
| Concorrência | 1 |
| Duração ativa reportada | aprox. 24,3 s |
| Tempo total decorrido | aprox. 28,0 s |
| Tokens de entrada, total | 785 |
| Tokens de saída, total | 781 |
| Latência média de requisição | 4,8 s |
| Latência mediana de requisição | 1,4 s |
| Latência P95 de requisição | 15,3 s |
| TTFT mediano | 123,5 ms |
| TTFT médio | 2.233,3 ms |
| TTFT P95 | 10.700,3 ms |
| ITL mediano | 16,8 ms |
| ITL médio | 16,7 ms |
| ITL P95 | 16,9 ms |
| TPOT mediano | 18,1 ms |
| TPOT médio | 30,9 ms |
| TPOT P95 | 55,9 ms |
| Throughput agregado reportado — requisições | aprox. 0,2 req/s |
| Throughput agregado reportado — entrada | aprox. 32,3 tokens/s |
| Throughput agregado reportado — saída | aprox. 32,2 tokens/s |
| Throughput agregado reportado — total | aprox. 64,5 tokens/s |

**Atenção à interpretação:** o resumo também mostrou uma média de tokens de saída por segundo de aproximadamente 87,5 em outra seção. Não se deve misturar métricas com denominadores ou definições diferentes: para comparações futuras, confira os campos e a semântica exata no CSV/JSON exportados. O valor de throughput agregado da tabela acima vem da seção de throughput do relatório do GuideLLM.

### 7.3 O que podemos concluir

- O backend OpenAI-compatible do Ollama respondeu às requisições do GuideLLM.
- O benchmark terminou sem erros nem requisições incompletas.
- A execução síncrona usou concorrência 1.
- A latência apresentou diferença considerável entre média e mediana, especialmente no TTFT. Uma hipótese é que a primeira requisição tenha incluído carregamento/aquecimento; é necessário inspecionar os registros individuais para confirmar.
- O pedido sintético de 128 tokens de saída não garante que todas as respostas terminem exatamente com 128 tokens. O relatório registrou mediana de saída de 80 tokens.
- Cinco requisições são uma amostra pequena. P95, média e dispersão são instáveis nessa amostra e não caracterizam ainda o desempenho sustentado do sistema.

Não foi registrado um monitoramento de `nvidia-smi` sincronizado com a execução do baseline. Portanto, o benchmark funcional e a evidência anterior de offload para GPU estão confirmados, mas não temos ainda uma série de utilização de GPU correlacionada a cada fase desse benchmark.

## 8. Comandos úteis para inspeção

### Estado do nó e recursos de GPU

```bash
minikube kubectl -p gpu-lab -- get nodes
minikube kubectl -p gpu-lab -- describe node gpu-lab
```

### Estado dos recursos do laboratório

```bash
minikube kubectl -p gpu-lab -- get all,pvc -n ml-lab
minikube kubectl -p gpu-lab -- logs deployment/ollama -n ml-lab
```

### API do Ollama pelo port-forward

```bash
curl http://localhost:11435/api/tags
curl http://localhost:11435/v1/models
```

### Ver uso da GPU durante uma carga

Em um terminal separado, antes de iniciar o benchmark:

```bash
watch -n 1 nvidia-smi
```

Isso permite observar utilização, memória da GPU e processos durante a carga. O monitoramento deve ser iniciado antes do benchmark e encerrado depois da execução.

### Inspecionar os resultados por requisição

```bash
cd ~/Documentos/ml-lab-benchmarks
ls -lh results/
head -n 5 results/baseline.csv
```

O CSV e o JSON devem ser preservados para comparar experimentos com a mesma configuração.

## 9. Manifestos e arquivos entregues com este relatório

- `manifest.yaml`: Namespace, PVC, Deployment e Service do Ollama, solicitação de GPU e Pod opcional de teste CUDA.
- `Dockerfile`: imagem do GuideLLM baseada em Python 3.13.
- Este relatório: arquitetura, procedimentos, comandos e interpretação do baseline.

O manifesto do Ollama não instala o NVIDIA Device Plugin: esse componente já foi instalado separadamente no cluster e é o responsável por anunciar `nvidia.com/gpu` ao scheduler.

Para reaplicar os recursos do arquivo:

```bash
minikube kubectl -p gpu-lab -- apply -f manifest.yaml
```

O Pod `gpu-test` incluído no arquivo é um teste de execução única com `restartPolicy: Never`. Depois de verificar seu resultado, ele pode terminar em estado `Completed`; isso é esperado.

## 10. Próxima etapa metodológica

Antes de aumentar concorrência ou carga, o próximo passo é inspecionar os registros individuais de `baseline.csv` para identificar qual requisição elevou a média e o P95 de TTFT e verificar os comprimentos reais de saída.

Depois disso, o experimento mais informativo é repetir a mesma configuração após uma requisição de aquecimento, mantendo constantes modelo, tokenizer, perfil, número de requisições e parâmetros sintéticos. Comparar o baseline frio com o baseline aquecido ajuda a separar custo de inicialização do comportamento em regime mais estável.

Somente depois dessa comparação vale avançar para cargas de maior concorrência. Cada rodada deve ser identificada e seus arquivos preservados, sem sobrescrever resultados anteriores.

## Referências técnicas

- GuideLLM — repositório: https://github.com/vllm-project/guidellm
- GuideLLM — guia de benchmark: https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md
- GuideLLM — backends: https://vllm-project.github.io/guidellm/stable/guides/backends/
- NVIDIA Kubernetes Device Plugin: https://github.com/NVIDIA/k8s-device-plugin
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
