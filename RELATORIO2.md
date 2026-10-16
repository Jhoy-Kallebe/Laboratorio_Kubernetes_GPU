# Relatório Técnico — Laboratório de Benchmark de Inferência com GuideLLM e Ollama
## Parte 2 — Diagnóstico, correção de incompatibilidade e estabelecimento do baseline

**Projeto:** MLOps — Laboratório de Inferência de LLMs  
**Data:** 16 de outubro de 2026  
**Status:** Concluído — baseline controlado estabelecido  
**Próxima etapa:** Benchmark de concorrência

---

# 1. Resumo executivo

Nesta segunda etapa do laboratório, nosso objetivo foi configurar e validar o **GuideLLM como ferramenta de benchmarking para modelos de linguagem executados localmente com Ollama**.

Utilizamos o modelo Qwen 2.5 3B, executado em uma NVIDIA RTX 3050 de 6 GB, com o Ollama implantado em um cluster Kubernetes gerenciado pelo Minikube.

Durante os experimentos, identificamos uma incompatibilidade entre o formato de requisição produzido pelo GuideLLM 0.8.1 e o comportamento da API OpenAI-compatible do Ollama.

Essa incompatibilidade fazia com que o modelo gerasse mais tokens do que o limite solicitado, comprometendo o controle das cargas de trabalho.

Após investigar o código-fonte do GuideLLM, desenvolvemos uma imagem Docker personalizada que corrigiu o problema.

**Principais resultados:**

- Identificação da incompatibilidade entre `max_completion_tokens` e `max_tokens`.
- Construção da imagem `guidellm-ollama:latest`.
- Validação do limite de geração de 128 tokens.
- Execução de um benchmark controlado com 30 requisições.
- Estabelecimento de métricas iniciais de TTFT, ITL, E2E e throughput.
- Identificação de fatores metodológicos que precisam ser considerados nos próximos experimentos.

O baseline apresentou **57,86 tokens de saída por segundo**, TTFT médio de **64,4 ms** e ITL médio de **16,7 ms**.

---

# 2. Arquitetura do laboratório

## 2.1 Infraestrutura utilizada

| Componente | Tecnologia |
|---|---|
| Sistema operacional | Ubuntu 26.04 |
| Processador | Intel Core i5 — 3ª geração |
| Memória RAM | 16 GB DDR3 |
| GPU | NVIDIA RTX 3050 6 GB |
| Orquestração | Kubernetes com Minikube |
| Containerização | Docker |
| Runtime de inferência | Ollama |
| Modelo | Qwen 2.5 3B |
| Ferramenta de benchmark | GuideLLM 0.8.1 |
| Interface de comunicação | OpenAI-compatible API |

O Ollama foi executado dentro do Kubernetes, enquanto o GuideLLM foi executado em um container Docker separado.

## 2.2 Arquitetura lógica

```text
                  Máquina física
              Ubuntu 26.04 Bare Metal
                         |
            +------------+------------+
            |                         |
            v                         v
     Docker Container             Minikube
            |                         |
            v                         v
        GuideLLM                 Kubernetes
            |                         |
            |                         v
            |                    Namespace ml-lab
            |                         |
            |                         v
            |                    Ollama Service
            |                         |
            |                         v
            |                    Ollama Pod
            |                         |
            |                         v
            |                    Qwen 2.5 3B
            |                         |
            |                         v
            |                    RTX 3050 GPU
            |                         |
            +-------------------------+
                    HTTP API
```

O acesso ao Ollama foi realizado por meio de um port-forward:

```bash
minikube kubectl -p gpu-lab -- \
  port-forward -n ml-lab service/ollama 11435:11434
```

O endpoint utilizado nos benchmarks foi:

```text
http://127.0.0.1:11435/v1/chat/completions
```

---

# 3. Problema identificado: controle da quantidade de tokens

## 3.1 Comportamento inicial

Configuramos o GuideLLM para gerar cargas sintéticas com:

```bash
--data kind=synthetic_text,prompt_tokens=128,output_tokens=128
```

Esperávamos que cada requisição produzisse, no máximo, 128 tokens de saída.

Entretanto, observamos resultados como:

| Requisição | Tokens gerados |
|---|---:|
| 1 | 120 |
| 2 | 177 |
| 3 | 152 |
| 4 | 727 |
| 5 | 92 |

O problema era evidente: algumas requisições ultrapassavam significativamente o limite configurado.

Isso introduzia uma variável não controlada no experimento.

Por exemplo, uma resposta de 727 tokens exige muito mais processamento do que uma resposta de 128 tokens.

Consequentemente, as métricas de latência e throughput deixavam de representar a carga originalmente planejada.

## 3.2 Investigação dos argumentos das requisições

Inspecionamos o arquivo JSON produzido pelo GuideLLM e identificamos que as requisições eram construídas utilizando:

```json
{
  "max_completion_tokens": 128,
  "ignore_eos": true,
  "stream": true
}
```

O parâmetro `max_completion_tokens` deveria estabelecer o limite de geração.

Entretanto, suspeitamos que o Ollama não estivesse interpretando corretamente esse campo.

---

# 4. Validação da incompatibilidade com a API do Ollama

Para verificar a hipótese, realizamos dois experimentos diretamente no endpoint OpenAI-compatible.

## 4.1 Experimento utilizando `max_tokens`

Enviamos uma requisição contendo:

```json
{
  "model": "qwen2.5:3b",
  "messages": [
    {
      "role": "user",
      "content": "Explique a arquitetura do Kubernetes."
    }
  ],
  "max_tokens": 16,
  "stream": false
}
```

O resultado apresentou:

```json
{
  "finish_reason": "length",
  "usage": {
    "completion_tokens": 16
  }
}
```

**Conclusão:** o Ollama respeitou o limite estabelecido por `max_tokens`.

## 4.2 Experimento utilizando `max_completion_tokens`

Em seguida, realizamos outro teste:

```json
{
  "model": "qwen2.5:3b",
  "messages": [
    {
      "role": "user",
      "content": "Explique detalhadamente como funciona a arquitetura do Kubernetes, incluindo seus principais componentes."
    }
  ],
  "max_completion_tokens": 16,
  "stream": false
}
```

O resultado foi:

```json
{
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 53,
    "completion_tokens": 1029,
    "total_tokens": 1082
  }
}
```

Apesar de solicitarmos apenas 16 tokens, o modelo gerou 1.029.

### Conclusão experimental

| Parâmetro | Limite solicitado | Tokens gerados | Resultado |
|---|---:|---:|---|
| `max_tokens` | 16 | 16 | Respeitado |
| `max_completion_tokens` | 16 | 1.029 | Não respeitado |

Esses resultados demonstraram que, **na versão do Ollama utilizada no laboratório, `max_completion_tokens` não estava limitando a geração**.

Isso forneceu uma explicação consistente para o comportamento observado no GuideLLM.

---

# 5. Investigação do código-fonte do GuideLLM

## 5.1 Localização do pacote

Identificamos o diretório de instalação do GuideLLM:

```text
/usr/local/lib/python3.13/site-packages/guidellm
```

Em seguida, pesquisamos as referências ao parâmetro:

```bash
grep -Rn \
  --include="*.py" \
  "max_completion_tokens" \
  /usr/local/lib/python3.13/site-packages/guidellm
```

Encontramos a implementação responsável pela construção das requisições no arquivo:

```text
guidellm/backends/openai/request_handlers.py
```

## 5.2 Código responsável pelo comportamento

O trecho identificado foi:

```python
# Handle output tokens
if data.output_metrics.text_tokens:
    arguments.body.update(
        {
            "max_completion_tokens": data.output_metrics.text_tokens,
            "stop": None,
            "ignore_eos": True,
        }
    )
elif kwargs.get("max_tokens") is not None:
    arguments.body["max_completion_tokens"] = kwargs["max_tokens"]
```

Esse código revelou dois comportamentos.

**Primeiro:** quando o dataset fornece uma quantidade de tokens de saída, o GuideLLM utiliza esse valor para preencher `max_completion_tokens`.

**Segundo:** mesmo quando configuramos `max_tokens` diretamente no backend, o GuideLLM utiliza esse valor para preencher `max_completion_tokens` no corpo da requisição.

Portanto, a configuração:

```bash
--backend ...,max_tokens=128
```

não resolvia o problema.

O GuideLLM continuava construindo requisições incompatíveis com o comportamento observado no Ollama.

---

# 6. Desenvolvimento da solução

## 6.1 Estratégia adotada

Consideramos a possibilidade de implementar um proxy HTTP para converter:

```text
max_completion_tokens
        |
        v
    max_tokens
```

Entretanto, essa solução adicionaria um componente ao caminho das requisições.

Em um laboratório de benchmarking, isso poderia introduzir latência adicional nas medições.

Por esse motivo, optamos por criar uma **imagem Docker personalizada do GuideLLM**.

## 6.2 Alteração implementada

Substituímos:

```python
"max_completion_tokens": data.output_metrics.text_tokens
```

Por:

```python
"max_tokens": data.output_metrics.text_tokens
```

Também substituímos:

```python
arguments.body["max_completion_tokens"] = kwargs["max_tokens"]
```

Por:

```python
arguments.body["max_tokens"] = kwargs["max_tokens"]
```

A alteração foi aplicada durante a construção de uma nova imagem Docker.

A imagem original permaneceu preservada.

| Imagem | Descrição |
|---|---|
| `guidellm-lab:latest` | GuideLLM original |
| `guidellm-ollama:latest` | GuideLLM adaptado para Ollama |

A correção foi específica para o formato de requisição utilizado no laboratório.

Não foi necessário modificar o Ollama nem adicionar serviços intermediários.

---

# 7. Validação da solução

Após construir a imagem personalizada, executamos cinco requisições sequenciais.

### Configuração

```text
Modelo: Qwen 2.5 3B
Input configurado: 128 tokens
Output máximo: 128 tokens
Concorrência: 1
Requisições: 5
Perfil: synchronous
```

### Resultados

| Indicador | Resultado |
|---|---:|
| Requisições bem-sucedidas | 5 |
| Erros | 0 |
| Output máximo observado | 128 tokens |
| Mediana de output | 128 tokens |
| P95 de output | 128 tokens |
| TTFT médio | 82,5 ms |
| ITL médio | 16,9 ms |
| E2E médio | 2,0 s |
| Throughput de saída | 57,14 tokens/s |

Nenhuma das cinco requisições ultrapassou o limite configurado.

**Conclusão:** a alteração resolveu o problema de controle do tamanho máximo das respostas no experimento de validação.

---

# 8. Benchmark controlado — 30 requisições

Após validar a correção, executamos um experimento com uma amostra maior.

## 8.1 Configuração

| Parâmetro | Valor |
|---|---|
| Runtime | Ollama |
| Modelo | Qwen 2.5 3B |
| Ferramenta | GuideLLM 0.8.1 modificado |
| Perfil | Synchronous |
| Concorrência | 1 |
| Requisições | 30 |
| Input configurado | 128 tokens |
| Output máximo | 128 tokens |
| Modelo aquecido | Sim |

Comando utilizado:

```bash
docker run --rm \
  --network host \
  -v "$PWD/results:/results" \
  guidellm-ollama:latest run \
  --backend kind=openai_http,target=http://127.0.0.1:11435,model=qwen2.5:3b,request_format=/v1/chat/completions,validate_backend=false \
  --tokenizer kind=huggingface_auto,model=Qwen/Qwen2.5-3B-Instruct \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=30 \
  --data kind=synthetic_text,prompt_tokens=128,output_tokens=128 \
  --output kind=json,path=/results/baseline-controlled.json \
  --output kind=csv,path=/results/baseline-controlled.csv
```

## 8.2 Resultados gerais

| Indicador | Resultado |
|---|---:|
| Requisições executadas | 30 |
| Requisições bem-sucedidas | 30 |
| Erros | 0 |
| Duração medida | 55,7 s |
| Tokens de entrada | 4.710 |
| Tokens de saída | 3.224 |
| Taxa de sucesso | 100% |

O benchmark foi concluído sem erros.

## 8.3 Latência

| Métrica | Média | Mediana | P95 |
|---|---:|---:|---:|
| E2E Latency | 1,9 s | 2,2 s | 2,3 s |
| TTFT | 64,4 ms | 62,0 ms | 116,9 ms |
| ITL | 16,7 ms | 16,7 ms | 17,2 ms |
| TPOT | 17,2 ms | 17,2 ms | 17,9 ms |

### Interpretação

**TTFT — Time to First Token**

Representa o tempo entre o envio da requisição e o recebimento do primeiro token.

O valor médio observado foi de 64,4 ms.

**ITL — Inter-Token Latency**

Representa o intervalo entre tokens consecutivos durante a geração.

O valor médio observado foi de 16,7 ms.

Podemos estimar a velocidade de geração:

\[
\frac{1000}{16,7}\approx59,9\text{ tokens/s}
\]

**E2E — End-to-End Latency**

Representa o tempo completo de atendimento da requisição.

A média foi de aproximadamente 1,9 segundo.

**TPOT — Time per Output Token**

Representa uma medida do tempo médio por token produzido, conforme a metodologia utilizada pelo GuideLLM.

O resultado médio foi de 17,2 ms.

## 8.4 Throughput

| Métrica | Resultado |
|---|---:|
| Requisições/s | 0,54 |
| Input tokens/s | 84,5 |
| Output tokens/s | 57,86 |
| Total tokens/s | 142,4 |

O resultado principal foi:

\[
\boxed{\text{Output Throughput}=57,86\text{ tokens/s}}
\]

Esse valor representa o throughput agregado observado durante o experimento com concorrência igual a 1.

Não deve ser interpretado como a capacidade máxima do runtime.

---

# 9. Análise da distribuição de tokens

Após executar o benchmark, investigamos a quantidade efetivamente gerada em cada requisição.

Obtivemos:

```text
Requisições: 30
Mínimo: 58
Máximo: 128
Média: 107.47
Com 128 tokens: 16
Abaixo de 128: 14
Acima de 128: 0
```

A distribuição foi:

```text
[128, 69, 128, 84, 128, 128, 128, 128, 128, 101,
 95, 81, 72, 128, 111, 128, 74, 123, 71, 128,
 63, 128, 107, 58, 128, 67, 128, 128, 128, 128]
```

## 9.1 Interpretação

| Categoria | Requisições | Percentual |
|---|---:|---:|
| Exatamente 128 tokens | 16 | 53,3% |
| Menos de 128 tokens | 14 | 46,7% |
| Mais de 128 tokens | 0 | 0% |

O limite máximo foi respeitado em todas as requisições.

Entretanto, aproximadamente 46,7% das respostas terminaram antes de atingir 128 tokens.

Isso demonstra que **controlar o limite máximo de geração não equivale a controlar exatamente o comprimento da resposta**.

Essa distinção será importante nos experimentos de concorrência.

---

# 10. Limitações metodológicas identificadas

Durante esta etapa, identificamos quatro fatores que precisam ser considerados nos próximos benchmarks.

### 10.1 Diferença entre tokens configurados e tokens efetivos

Embora o dataset tenha sido configurado com 128 tokens de entrada, o GuideLLM registrou 157 tokens por requisição.

Isso pode estar relacionado à formatação das mensagens e aos tokens introduzidos pelo template de chat.

Devemos registrar ambos os valores.

### 10.2 Variação no comprimento das respostas

As respostas variaram entre 58 e 128 tokens.

Essa diferença influencia diretamente a latência E2E e o throughput observado.

Por isso, a distribuição dos tokens gerados deve acompanhar os resultados dos próximos experimentos.

### 10.3 Comportamento de `ignore_eos`

O GuideLLM utiliza:

```json
{
  "ignore_eos": true
}
```

Ainda não verificamos experimentalmente se o Ollama reconhece esse parâmetro.

Consequentemente, o motivo exato das interrupções antecipadas permanece em aberto.

### 10.4 Possível gargalo na geração dos dados

O GuideLLM apresentou o aviso:

```text
mean generation delay was 1% of mean time to first token
(0.001s / 0.064s; threshold 1%).

This means that the benchmark was likely bottlenecked
by the data generation.
```

O atraso médio reportado foi de aproximadamente 1 ms.

O aviso não comprova que houve um gargalo significativo, mas indica que devemos monitorar o overhead do cliente, especialmente nos experimentos com maior concorrência.

---

# 11. Preparação para os testes de concorrência

Com o baseline estabelecido, definimos a próxima etapa: avaliar como o Ollama responde ao aumento da quantidade de requisições simultâneas.

Os níveis inicialmente planejados são:

| Experimento | Concorrência | Input configurado | Output máximo |
|---|---:|---:|---:|
| C1 | 1 | 128 | 128 |
| C2 | 2 | 128 | 128 |
| C4 | 4 | 128 | 128 |
| C8 | 8 | 128 | 128 |

O objetivo será analisar a relação entre:

\[
\text{Concorrência}\rightarrow
\begin{cases}
\text{TTFT}\\
\text{ITL}\\
\text{E2E}\\
\text{Throughput}
\end{cases}
\]

Entretanto, nossa implantação atual do Ollama utiliza:

```text
OLLAMA_NUM_PARALLEL=1
```

Isso significa que a concorrência configurada no cliente não representa necessariamente paralelismo efetivo na GPU.

Assim, os primeiros experimentos avaliarão principalmente a capacidade de atendimento do servidor diante de múltiplas requisições em andamento, incluindo possíveis efeitos de enfileiramento.

Posteriormente, poderemos estudar o impacto de modificar o paralelismo interno do Ollama.

---

# 12. Conclusões

Nesta segunda etapa do laboratório, conseguimos transformar uma configuração inicial de benchmark em um ambiente com maior controle experimental.

A investigação demonstrou a importância de validar não apenas as métricas produzidas pela ferramenta, mas também **as requisições que ela realmente prepara e o comportamento do runtime ao recebê-las**.

O problema identificado com `max_completion_tokens` poderia produzir resultados incorretamente interpretados em uma comparação entre diferentes runtimes.

A correção permitiu controlar o limite máximo de geração sem introduzir um proxy HTTP intermediário.

Também estabelecemos nosso primeiro baseline:

| Métrica | Resultado de referência |
|---|---:|
| Modelo | Qwen 2.5 3B |
| GPU | RTX 3050 6 GB |
| Concorrência | 1 |
| Requisições | 30 |
| TTFT médio | **64,4 ms** |
| ITL médio | **16,7 ms** |
| E2E médio | **1,9 s** |
| Throughput de saída | **57,86 tokens/s** |
| Taxa de sucesso | **100%** |

Esses resultados constituem uma referência inicial para a configuração estudada.

Ainda não representam a capacidade máxima do sistema nem permitem concluir como ele se comportará sob cargas concorrentes.

## 13. Próxima etapa

O próximo objetivo do laboratório será realizar benchmarks com concorrência crescente, começando por duas requisições simultâneas.

Precisaremos comparar os resultados com o baseline, observando especialmente o aumento da latência, a evolução do throughput agregado e a distribuição real dos tokens gerados.

Também será necessário distinguir os efeitos do enfileiramento daqueles relacionados ao paralelismo interno do runtime.

**Status final da Parte 2: concluída.**

**Resultado principal:** GuideLLM adaptado e validado para benchmarking controlado com Ollama, com baseline de 30 requisições estabelecido.