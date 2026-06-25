#!/usr/bin/env bash
# ============================================================================
#  run_experiments.sh
#  Executa o AG paralelo variando (numero de threads) x (tamanho da entrada),
#  repetindo REPS vezes cada configuracao, e grava todos os tempos em CSV.
#
#  Saida: results/timings.csv com colunas:
#    pop,dim,gen,threads,load,seed,best_fitness,tempo_s,size_label,run
#
#  Uso:  bash scripts/run_experiments.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # raiz do projeto

BIN=./ga_rastrigin
OUT=results/timings.csv
REPS=10                     # repeticoes por configuracao (media + IC 95%)
GEN=30                      # geracoes (fixo entre experimentos)
LOAD=50                     # carga artificial por avaliacao (alto custo)

# Numeros de threads a testar (1..12 cobre P-cores, E-cores e HT do i5-12450H)
THREADS=(1 2 4 6 8 10 12)

# Tamanhos de entrada: rotulo "pop dim"
# Cada linha = um cenario de tamanho (pequeno, medio, grande)
declare -a SIZES=(
  "P  800   200"
  "M  1500  300"
  "G  2500  400"
)

# Garante binario compilado
if [[ ! -x "$BIN" ]]; then
  echo ">> Compilando..."
  make
fi

mkdir -p results
echo "pop,dim,gen,threads,load,seed,best_fitness,tempo_s,size_label,run" > "$OUT"

total=$(( ${#SIZES[@]} * ${#THREADS[@]} * REPS ))
count=0

for size in "${SIZES[@]}"; do
  read -r LABEL POP DIM <<< "$size"
  for T in "${THREADS[@]}"; do
    for ((r=1; r<=REPS; r++)); do
      count=$((count+1))
      SEED=$(( 1000 + r ))   # semente varia por repeticao
      line=$("$BIN" --pop "$POP" --dim "$DIM" --gen "$GEN" \
                    --threads "$T" --load "$LOAD" --seed "$SEED" --csv)
      echo "${line},${LABEL},${r}" >> "$OUT"
      printf "\r[%3d/%3d] tamanho=%s threads=%2d run=%2d  " \
             "$count" "$total" "$LABEL" "$T" "$r"
    done
  done
done

echo ""
echo ">> Concluido. Resultados em: $OUT"
