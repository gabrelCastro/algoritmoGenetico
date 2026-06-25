#!/usr/bin/env python3
# ============================================================================
#  plot_results.py
#  Le results/timings.csv e gera:
#    - results/tempo.png       : tempo medio de execucao com IC 95% (10 exec.)
#    - results/speedup.png     : speedup S(p) = T(1)/T(p)
#    - results/eficiencia.png  : eficiencia E(p) = S(p)/p
#    - results/karpflatt.png   : metrica de Karp-Flatt
#    - results/metricas.csv    : tabela consolidada (para o artigo)
#
#  Uso:  .venv/bin/python scripts/plot_results.py
# ============================================================================
import os
import csv
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
CSV_IN = os.path.join(RESULTS, "timings.csv")

CONF = 0.95  # nivel de confianca dos intervalos


def load():
    """Le o CSV e agrupa os tempos por (size_label, threads)."""
    data = defaultdict(lambda: defaultdict(list))  # size -> threads -> [tempos]
    meta = {}  # size -> (pop, dim)
    with open(CSV_IN, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = row["size_label"]
            thr = int(row["threads"])
            data[size][thr].append(float(row["tempo_s"]))
            meta[size] = (int(row["pop"]), int(row["dim"]))
    return data, meta


def ci95(samples):
    """Media e meia-largura do IC 95% (t-Student)."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    mean = arr.mean()
    if n < 2:
        return mean, 0.0
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + CONF) / 2.0, n - 1)
    return mean, h


# Rotulos amigaveis para os tamanhos de entrada
LABELS = {"P": "Pequena", "M": "Media", "G": "Grande"}


def main():
    data, meta = load()
    sizes = sorted(data.keys(), key=lambda s: meta[s][0] * meta[s][1])

    # ---- consolida estatisticas ------------------------------------------
    stats_by_size = {}
    for size in sizes:
        threads = sorted(data[size].keys())
        means, errs = {}, {}
        for t in threads:
            m, h = ci95(data[size][t])
            means[t] = m
            errs[t] = h
        t1 = means[min(threads)]  # baseline = menor numero de threads (=1)
        rows = []
        for t in threads:
            speedup = t1 / means[t]
            eff = speedup / t
            # Karp-Flatt: e = (1/S - 1/p) / (1 - 1/p), indefinido em p=1
            if t == 1:
                kf = float("nan")
            else:
                kf = (1.0 / speedup - 1.0 / t) / (1.0 - 1.0 / t)
            rows.append((t, means[t], errs[t], speedup, eff, kf))
        stats_by_size[size] = (threads, means, errs, rows)

    # ---- grava tabela consolidada ----------------------------------------
    out_csv = os.path.join(RESULTS, "metricas.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tamanho", "pop", "dim", "threads", "tempo_medio_s",
                    "ic95_s", "speedup", "eficiencia", "karp_flatt"])
        for size in sizes:
            pop, dim = meta[size]
            _, _, _, rows = stats_by_size[size]
            for (t, m, h, sp, ef, kf) in rows:
                w.writerow([size, pop, dim, t, f"{m:.6f}", f"{h:.6f}",
                            f"{sp:.4f}", f"{ef:.4f}",
                            "" if math.isnan(kf) else f"{kf:.4f}"])
    print(f">> Tabela consolidada: {out_csv}")

    markers = ["o", "s", "^", "D"]

    # ---- 1) tempo de execucao com IC 95% ---------------------------------
    plt.figure(figsize=(7, 5))
    for i, size in enumerate(sizes):
        threads, means, errs, _ = stats_by_size[size]
        pop, dim = meta[size]
        y = [means[t] for t in threads]
        e = [errs[t] for t in threads]
        plt.errorbar(threads, y, yerr=e, marker=markers[i % len(markers)],
                     capsize=4, label=f"{LABELS.get(size, size)} (pop={pop}, dim={dim})")
    plt.xlabel("Numero de threads")
    plt.ylabel("Tempo de execucao (s)")
    plt.title("Tempo de execucao (media de 10 execucoes, IC 95%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "tempo.png"), dpi=150)
    plt.close()

    # ---- 2) speedup -------------------------------------------------------
    plt.figure(figsize=(7, 5))
    maxt = max(max(stats_by_size[s][0]) for s in sizes)
    plt.plot([1, maxt], [1, maxt], "k--", alpha=0.5, label="Speedup ideal (linear)")
    for i, size in enumerate(sizes):
        threads, _, _, rows = stats_by_size[size]
        pop, dim = meta[size]
        sp = [r[3] for r in rows]
        plt.plot(threads, sp, marker=markers[i % len(markers)],
                 label=f"{LABELS.get(size, size)} (pop={pop}, dim={dim})")
    plt.xlabel("Numero de threads")
    plt.ylabel("Speedup  S(p) = T(1)/T(p)")
    plt.title("Speedup x numero de threads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "speedup.png"), dpi=150)
    plt.close()

    # ---- 3) eficiencia ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Eficiencia ideal (1,0)")
    for i, size in enumerate(sizes):
        threads, _, _, rows = stats_by_size[size]
        pop, dim = meta[size]
        ef = [r[4] for r in rows]
        plt.plot(threads, ef, marker=markers[i % len(markers)],
                 label=f"{LABELS.get(size, size)} (pop={pop}, dim={dim})")
    plt.xlabel("Numero de threads")
    plt.ylabel("Eficiencia  E(p) = S(p)/p")
    plt.title("Eficiencia x numero de threads")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eficiencia.png"), dpi=150)
    plt.close()

    # ---- 4) Karp-Flatt ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    for i, size in enumerate(sizes):
        threads, _, _, rows = stats_by_size[size]
        pop, dim = meta[size]
        xs = [r[0] for r in rows if not math.isnan(r[5])]
        kf = [r[5] for r in rows if not math.isnan(r[5])]
        plt.plot(xs, kf, marker=markers[i % len(markers)],
                 label=f"{LABELS.get(size, size)} (pop={pop}, dim={dim})")
    plt.xlabel("Numero de threads")
    plt.ylabel("Fracao serial experimental  e  (Karp-Flatt)")
    plt.title("Metrica de Karp-Flatt x numero de threads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "karpflatt.png"), dpi=150)
    plt.close()

    print(">> Graficos gerados em results/: tempo.png, speedup.png, "
          "eficiencia.png, karpflatt.png")

    # ---- resumo no terminal ----------------------------------------------
    for size in sizes:
        pop, dim = meta[size]
        _, _, _, rows = stats_by_size[size]
        print(f"\n=== Tamanho {LABELS.get(size, size)} (pop={pop}, dim={dim}) ===")
        print(f"{'thr':>4} {'tempo(s)':>10} {'+/-IC95':>9} {'speedup':>8} "
              f"{'efic.':>7} {'karpflatt':>10}")
        for (t, m, h, sp, ef, kf) in rows:
            kfs = "   -   " if math.isnan(kf) else f"{kf:10.4f}"
            print(f"{t:>4} {m:>10.4f} {h:>9.4f} {sp:>8.3f} {ef:>7.3f} {kfs}")


if __name__ == "__main__":
    main()
