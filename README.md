# Algoritmo Genético Paralelo (OpenMP) — Maximização da Função Rastrigin

Trabalho da disciplina de **Programação Paralela**. Implementa um **Algoritmo
Genético (AG)** em C++ que maximiza uma função matemática de alto custo
computacional, **paralelizando a etapa de avaliação de aptidão (fitness)** com
**OpenMP**, e analisa as métricas de desempenho **Speedup**, **Eficiência** e
**Karp-Flatt**.

## Problema

Maximizar o negativo da função de **Rastrigin** (função clássica de
minimização, com muitos mínimos locais e ótimo global conhecido em
`x = (0,...,0)`, onde `f = 0`):

```
Rastrigin(x) = A*n + sum_i [ x_i^2 - A*cos(2*pi*x_i) ],   A = 10
fitness(x)   = -Rastrigin(x)            (máximo global = 0)
```

A avaliação de cada indivíduo recebe uma **carga artificial** (`--load`) que
simula uma função-objetivo cara (ex.: uma simulação física). Como cada
indivíduo é avaliado de forma independente, a avaliação da população é
*embaraçosamente paralela*.

## Estrutura

```
src/ga_rastrigin.cpp     # AG paralelo com OpenMP
Makefile                 # compila com -O2 -fopenmp
scripts/run_experiments.sh   # roda 10x cada (threads x tamanho) -> results/timings.csv
scripts/plot_results.py      # gera gráficos (IC 95%, speedup, eficiência, Karp-Flatt)
results/                 # CSVs e PNGs gerados
artigo_final.pdf         # artigo final compilado
apresentacao_roteiro.txt # roteiro para a apresentação do projeto
```

## Pré-requisitos

- `g++` com suporte a OpenMP (testado com g++ 13.3)
- Python 3 com `numpy`, `matplotlib`, `scipy` (para os gráficos)

## Como compilar

```bash
make
```

## Como executar (uma rodada)

```bash
./ga_rastrigin --pop 2000 --dim 500 --gen 200 --threads 8 --load 200 --verbose
```

Argumentos:

| Flag        | Significado                                   | Padrão |
|-------------|-----------------------------------------------|--------|
| `--pop`     | tamanho da população                          | 2000   |
| `--dim`     | dimensão (nº de variáveis) do indivíduo       | 500    |
| `--gen`     | número de gerações                            | 200    |
| `--threads` | nº de threads OpenMP                           | 1      |
| `--load`    | fator de carga artificial por avaliação        | 200    |
| `--seed`    | semente do gerador aleatório                  | 42     |
| `--pmut`    | probabilidade de mutação por gene             | 0.05   |
| `--pcross`  | probabilidade de crossover                    | 0.9    |
| `--csv`     | imprime uma linha CSV (para experimentos)      | off    |
| `--verbose` | imprime o progresso por geração               | off    |

O melhor fitness é **idêntico** para qualquer número de threads (mesma semente),
comprovando que a paralelização não introduz condição de corrida.

## Reproduzir os experimentos e gráficos

```bash
# 1) roda a bateria completa (threads x tamanho, 10 execuções cada)
bash scripts/run_experiments.sh

# 2) instala dependências de plotagem (uma vez), via virtualenv
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib scipy

# 3) gera gráficos e a tabela de métricas
.venv/bin/python scripts/plot_results.py
```

Saídas em `results/`: `timings.csv` (tempos brutos), `metricas.csv` (tabela
consolidada) e os gráficos `tempo.png`, `speedup.png`, `eficiencia.png`,
`karpflatt.png`.

## Métricas

- **Speedup:** `S(p) = T(1) / T(p)`
- **Eficiência:** `E(p) = S(p) / p`
- **Karp-Flatt:** `e = (1/S(p) - 1/p) / (1 - 1/p)` — fração serial experimental

O artigo compilado no formato SBC está disponível em `artigo_final.pdf` na raiz do projeto.
