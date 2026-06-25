// ============================================================================
//  Algoritmo Genetico Paralelo (OpenMP) para maximizacao da funcao Rastrigin
// ----------------------------------------------------------------------------
//  Problema: maximizar uma funcao matematica de alto custo computacional.
//
//  Otimizamos o NEGATIVO da funcao de Rastrigin (funcao classica de
//  minimizacao). Maximizar -Rastrigin equivale a minimizar Rastrigin, cujo
//  otimo global e f(x) = 0 em x = (0, 0, ..., 0).
//
//  A etapa paralelizada e EXCLUSIVAMENTE a avaliacao de aptidao (fitness) da
//  populacao, que e "embaracosamente paralela": cada individuo e avaliado de
//  forma independente. As demais etapas (selecao, crossover, mutacao) sao
//  sequenciais de proposito, constituindo a fracao serial do programa
//  (relevante para a metrica de Karp-Flatt).
//
//  Compilacao:  g++ -O2 -fopenmp -o ga_rastrigin src/ga_rastrigin.cpp
//
//  Uso:
//    ./ga_rastrigin --pop 2000 --dim 500 --gen 200 --threads 8
//                   --load 200 --seed 42 [--csv] [--verbose]
//
//  Disciplina: Programacao Paralela / Algoritmos Geneticos
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <string>
#include <omp.h>

// ---------------------------------------------------------------------------
// Parametros do problema e do AG (com valores padrao)
// ---------------------------------------------------------------------------
struct Params {
    int    pop      = 2000;   // tamanho da populacao
    int    dim      = 500;    // dimensao do vetor (numero de variaveis)
    int    gen       = 200;   // numero de geracoes
    int    threads   = 1;     // numero de threads OpenMP
    int    load      = 200;   // fator de carga artificial por avaliacao
    unsigned seed    = 42;    // semente do gerador aleatorio
    double lo        = -5.12; // limite inferior do dominio (padrao Rastrigin)
    double hi        =  5.12; // limite superior do dominio
    double pcross    = 0.9;   // probabilidade de crossover
    double pmut      = 0.05;  // probabilidade de mutacao por gene
    double sigma     = 0.30;  // desvio padrao da mutacao gaussiana
    int    tour      = 3;     // tamanho do torneio de selecao
    bool   csv       = false; // saida em formato CSV (1 linha)
    bool   verbose   = false; // imprime progresso por geracao
};

// ---------------------------------------------------------------------------
// Funcao de Rastrigin com CARGA ARTIFICIAL para simular alto custo.
//
//   Rastrigin(x) = A*n + sum_i [ x_i^2 - A*cos(2*pi*x_i) ],  A = 10
//
//   Retornamos -Rastrigin(x) para transformar em maximizacao.
//
//   O parametro 'load' injeta um laco extra de trabalho trigonometrico por
//   avaliacao, simulando uma funcao-objetivo cara (ex.: uma simulacao fisica).
//   Sem essa carga, a avaliacao seria rapida demais e o overhead das threads
//   dominaria, escondendo o ganho da paralelizacao.
// ---------------------------------------------------------------------------
static inline double fitness(const double* x, int dim, int load) {
    const double A  = 10.0;
    const double PI = 3.14159265358979323846;
    double sum = A * dim;
    for (int i = 0; i < dim; ++i) {
        sum += x[i] * x[i] - A * std::cos(2.0 * PI * x[i]);
    }

    // Carga artificial: trabalho adicional deterministico que depende de x,
    // mas NAO altera o resultado (somamos e subtraimos a mesma quantidade).
    if (load > 0) {
        double extra = 0.0;
        for (int k = 0; k < load; ++k) {
            for (int i = 0; i < dim; ++i) {
                extra += std::sin(x[i]) * std::cos(x[i]);
            }
        }
        sum += extra - extra; // anula o efeito numerico, mantendo o custo
    }

    return -sum; // maximizacao
}

// ---------------------------------------------------------------------------
// Selecao por torneio: sorteia 'tour' individuos e devolve o indice do melhor.
// ---------------------------------------------------------------------------
static inline int tournament(const std::vector<double>& fit, int tour,
                             std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, (int)fit.size() - 1);
    int best = pick(rng);
    for (int t = 1; t < tour; ++t) {
        int c = pick(rng);
        if (fit[c] > fit[best]) best = c;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Leitura dos argumentos de linha de comando
// ---------------------------------------------------------------------------
static void parse_args(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Faltou valor para %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--pop"))     p.pop     = std::atoi(next("--pop"));
        else if (!std::strcmp(argv[i], "--dim"))     p.dim     = std::atoi(next("--dim"));
        else if (!std::strcmp(argv[i], "--gen"))     p.gen     = std::atoi(next("--gen"));
        else if (!std::strcmp(argv[i], "--threads")) p.threads = std::atoi(next("--threads"));
        else if (!std::strcmp(argv[i], "--load"))    p.load    = std::atoi(next("--load"));
        else if (!std::strcmp(argv[i], "--seed"))    p.seed    = (unsigned)std::strtoul(next("--seed"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--pmut"))    p.pmut    = std::atof(next("--pmut"));
        else if (!std::strcmp(argv[i], "--pcross"))  p.pcross  = std::atof(next("--pcross"));
        else if (!std::strcmp(argv[i], "--csv"))     p.csv     = true;
        else if (!std::strcmp(argv[i], "--verbose")) p.verbose = true;
        else {
            std::fprintf(stderr, "Argumento desconhecido: %s\n", argv[i]);
            std::exit(1);
        }
    }
}

int main(int argc, char** argv) {
    Params p;
    parse_args(argc, argv, p);
    omp_set_num_threads(p.threads);

    // Gerador principal (operadores sequenciais usam este RNG)
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> uni(p.lo, p.hi);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double>       gauss(0.0, p.sigma);

    // -----------------------------------------------------------------------
    // Inicializacao da populacao (matriz linearizada pop x dim)
    // -----------------------------------------------------------------------
    std::vector<double> X(  (size_t)p.pop * p.dim);
    std::vector<double> Xn( (size_t)p.pop * p.dim); // proxima geracao
    std::vector<double> fit(p.pop);
    for (size_t k = 0; k < X.size(); ++k) X[k] = uni(rng);

    // -----------------------------------------------------------------------
    // Cronometro: medimos APENAS o laco evolutivo (nao a alocacao)
    // -----------------------------------------------------------------------
    double t0 = omp_get_wtime();

    double best_fit = -std::numeric_limits<double>::infinity();
    std::vector<double> best_ind(p.dim);

    for (int g = 0; g < p.gen; ++g) {
        // ---- ETAPA PARALELA: avaliacao de aptidao da populacao -----------
        // Cada individuo e avaliado de forma independente -> paralelizavel.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < p.pop; ++i) {
            fit[i] = fitness(&X[(size_t)i * p.dim], p.dim, p.load);
        }

        // ---- ETAPA SEQUENCIAL: elitismo (encontra o melhor) --------------
        int best_idx = 0;
        for (int i = 1; i < p.pop; ++i)
            if (fit[i] > fit[best_idx]) best_idx = i;
        if (fit[best_idx] > best_fit) {
            best_fit = fit[best_idx];
            std::copy(&X[(size_t)best_idx * p.dim],
                      &X[(size_t)best_idx * p.dim] + p.dim, best_ind.begin());
        }

        if (p.verbose && (g % 20 == 0 || g == p.gen - 1))
            std::fprintf(stderr, "[gen %4d] melhor_fitness = %.6f\n", g, best_fit);

        // ---- ETAPA SEQUENCIAL: nova geracao (selecao+crossover+mutacao) --
        // Elitismo: o melhor individuo passa intacto para a posicao 0.
        std::copy(best_ind.begin(), best_ind.end(), &Xn[0]);

        for (int i = 1; i < p.pop; ++i) {
            int a = tournament(fit, p.tour, rng);
            int b = tournament(fit, p.tour, rng);
            const double* pa = &X[(size_t)a * p.dim];
            const double* pb = &X[(size_t)b * p.dim];
            double* child    = &Xn[(size_t)i * p.dim];

            // Crossover aritmetico (blend) gene a gene
            if (unit(rng) < p.pcross) {
                for (int d = 0; d < p.dim; ++d) {
                    double alpha = unit(rng);
                    child[d] = alpha * pa[d] + (1.0 - alpha) * pb[d];
                }
            } else {
                std::copy(pa, pa + p.dim, child); // sem crossover: clona pai a
            }

            // Mutacao gaussiana com clamp ao dominio
            for (int d = 0; d < p.dim; ++d) {
                if (unit(rng) < p.pmut) {
                    child[d] += gauss(rng);
                    if (child[d] < p.lo) child[d] = p.lo;
                    if (child[d] > p.hi) child[d] = p.hi;
                }
            }
        }

        X.swap(Xn); // a nova geracao vira a populacao atual
    }

    double elapsed = omp_get_wtime() - t0;

    // -----------------------------------------------------------------------
    // Saida
    // -----------------------------------------------------------------------
    if (p.csv) {
        // pop,dim,gen,threads,load,seed,best_fitness,tempo_s
        std::printf("%d,%d,%d,%d,%d,%u,%.6f,%.6f\n",
                    p.pop, p.dim, p.gen, p.threads, p.load, p.seed,
                    best_fit, elapsed);
    } else {
        std::printf("==================================================\n");
        std::printf(" AG Paralelo - Rastrigin (maximizacao de -f)\n");
        std::printf("--------------------------------------------------\n");
        std::printf(" pop=%d dim=%d gen=%d threads=%d load=%d seed=%u\n",
                    p.pop, p.dim, p.gen, p.threads, p.load, p.seed);
        std::printf(" melhor fitness  : %.6f  (otimo = 0)\n", best_fit);
        std::printf(" tempo de exec.  : %.4f s\n", elapsed);
        std::printf("==================================================\n");
    }
    return 0;
}
