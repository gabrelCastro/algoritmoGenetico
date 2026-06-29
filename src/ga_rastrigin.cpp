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
//  sequenciais de proposito.
//
//  Compilacao:  g++ -O2 -fopenmp -o ga_rastrigin src/ga_rastrigin.cpp
//
//  Disciplina: Programacao Paralela / Algoritmos Geneticos
// ============================================================================

#include <cstdio>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <string>

// ---------------------------------------------------------------------------
// Estruturas de Dados
// ---------------------------------------------------------------------------

// Representa um individuo da populacao
struct Individuo {
    std::vector<double> genes;
    double fitness;
};

// Parametros do problema e do AG (com valores padrao)
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
    bool   csv       = false; // saida em formato CSV
    bool   verbose   = false; // imprime progresso por geracao
};

// ---------------------------------------------------------------------------
// Funcoes Auxiliares e Operadores Genéticos
// ---------------------------------------------------------------------------

// Carga artificial: simula um trabalho pesado para que a avaliacao demore
// mais e justifique o uso do paralelismo. Nao altera o resultado real.
void simularTrabalhoPesado(const std::vector<double>& genes, int load) {
    if (load <= 0) return;
    double extra = 0.0;
    for (int k = 0; k < load; ++k) {
        for (double gene : genes) {
            extra += std::sin(gene) * std::cos(gene);
        }
    }
}

// Avalia um individuo usando a funcao Rastrigin
double calcularFitness(const std::vector<double>& genes, int load) {
    const double A  = 10.0;
    const double PI = 3.14159265358979323846;
    double sum = A * genes.size();
    
    for (double gene : genes) {
        sum += gene * gene - A * std::cos(2.0 * PI * gene);
    }

    simularTrabalhoPesado(genes, load);

    return -sum; // Retorna negativo pois queremos MAXIMIZAR
}

// Selecao por torneio
int selecaoPorTorneio(const std::vector<Individuo>& populacao, int tamanhoTorneio, std::mt19937& rng) {
    std::uniform_int_distribution<int> distrib(0, populacao.size() - 1);
    int melhorIndice = distrib(rng);
    
    for (int t = 1; t < tamanhoTorneio; ++t) {
        int competidor = distrib(rng);
        if (populacao[competidor].fitness > populacao[melhorIndice].fitness) {
            melhorIndice = competidor;
        }
    }
    return melhorIndice;
}

// Crossover Aritmetico (Blend)
Individuo cruzamentoAritmetico(const Individuo& pai1, const Individuo& pai2, double pcross, std::mt19937& rng) {
    Individuo filho;
    filho.genes.resize(pai1.genes.size());
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    
    if (chance(rng) < pcross) {
        for (size_t i = 0; i < pai1.genes.size(); ++i) {
            double alpha = chance(rng);
            filho.genes[i] = alpha * pai1.genes[i] + (1.0 - alpha) * pai2.genes[i];
        }
    } else {
        filho.genes = pai1.genes; // Clona pai1 se nao ocorrer crossover
    }
    return filho;
}

// Mutacao Gaussiana
void mutacaoGaussiana(Individuo& ind, double pmut, double sigma, double limiteInf, double limiteSup, std::mt19937& rng) {
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, sigma);
    
    for (size_t i = 0; i < ind.genes.size(); ++i) {
        if (chance(rng) < pmut) {
            ind.genes[i] += gauss(rng);
            // Mantem dentro dos limites
            if (ind.genes[i] < limiteInf) ind.genes[i] = limiteInf;
            if (ind.genes[i] > limiteSup) ind.genes[i] = limiteSup;
        }
    }
}

// ---------------------------------------------------------------------------
// Leitura dos argumentos simplificada
// ---------------------------------------------------------------------------
void lerArgumentos(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--pop") p.pop = std::stoi(argv[++i]);
            else if (arg == "--dim") p.dim = std::stoi(argv[++i]);
            else if (arg == "--gen") p.gen = std::stoi(argv[++i]);
            else if (arg == "--threads") p.threads = std::stoi(argv[++i]);
            else if (arg == "--load") p.load = std::stoi(argv[++i]);
            else if (arg == "--seed") p.seed = std::stoul(argv[++i]);
            else if (arg == "--pmut") p.pmut = std::stod(argv[++i]);
            else if (arg == "--pcross") p.pcross = std::stod(argv[++i]);
        }
        if (arg == "--csv") p.csv = true;
        if (arg == "--verbose") p.verbose = true;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Params p;
    lerArgumentos(argc, argv, p);
    omp_set_num_threads(p.threads);

    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> distSorteioInicial(p.lo, p.hi);

    // Inicializacao da populacao
    std::vector<Individuo> populacao(p.pop);
    std::vector<Individuo> novaPopulacao(p.pop);
    
    for (int i = 0; i < p.pop; ++i) {
        populacao[i].genes.resize(p.dim);
        for (int j = 0; j < p.dim; ++j) {
            populacao[i].genes[j] = distSorteioInicial(rng);
        }
    }

    double melhorFitnessGlobal = -1e9; // valor inicial bem pequeno

    double tempoInicio = omp_get_wtime();

    // Laco Evolutivo
    for (int geracao = 0; geracao < p.gen; ++geracao) {
        
        // ===================================================================
        // ETAPA PARALELA: Avaliacao de Aptidao (Fitness)
        // Cada individuo e avaliado de forma independente (Embaracosamente Paralelo)
        // ===================================================================
        #pragma omp parallel for
        for (int i = 0; i < p.pop; ++i) {
            populacao[i].fitness = calcularFitness(populacao[i].genes, p.load);
        }

        // ===================================================================
        // ETAPA SEQUENCIAL: Elitismo e Criacao da Nova Geracao
        // ===================================================================
        
        // 1. Acha o melhor individuo desta geracao
        int indiceMelhor = 0;
        for (int i = 1; i < p.pop; ++i) {
            if (populacao[i].fitness > populacao[indiceMelhor].fitness) {
                indiceMelhor = i;
            }
        }
        
        if (populacao[indiceMelhor].fitness > melhorFitnessGlobal) {
            melhorFitnessGlobal = populacao[indiceMelhor].fitness;
        }

        if (p.verbose && (geracao % 20 == 0 || geracao == p.gen - 1)) {
            printf("[Geracao %4d] Melhor Fitness = %.6f\n", geracao, melhorFitnessGlobal);
        }

        // 2. O melhor individuo passa para a proxima geracao intacto (Elitismo)
        novaPopulacao[0] = populacao[indiceMelhor];

        // 3. Preenche o resto da nova populacao
        for (int i = 1; i < p.pop; ++i) {
            int indicePai1 = selecaoPorTorneio(populacao, p.tour, rng);
            int indicePai2 = selecaoPorTorneio(populacao, p.tour, rng);
            
            // Cruzamento
            novaPopulacao[i] = cruzamentoAritmetico(populacao[indicePai1], populacao[indicePai2], p.pcross, rng);
            
            // Mutacao
            mutacaoGaussiana(novaPopulacao[i], p.pmut, p.sigma, p.lo, p.hi, rng);
        }

        // Substitui a populacao antiga pela nova
        populacao = novaPopulacao;
    }

    double tempoExecucao = omp_get_wtime() - tempoInicio;

    // Resultados
    if (p.csv) {
        printf("%d,%d,%d,%d,%d,%u,%.6f,%.6f\n",
                    p.pop, p.dim, p.gen, p.threads, p.load, p.seed,
                    melhorFitnessGlobal, tempoExecucao);
    } else {
        printf("==================================================\n");
        printf(" AG Paralelo - Rastrigin (maximizacao de -f)\n");
        printf("--------------------------------------------------\n");
        printf(" pop=%d dim=%d gen=%d threads=%d load=%d seed=%u\n",
                    p.pop, p.dim, p.gen, p.threads, p.load, p.seed);
        printf(" melhor fitness  : %.6f  (otimo = 0)\n", melhorFitnessGlobal);
        printf(" tempo de exec.  : %.4f s\n", tempoExecucao);
        printf("==================================================\n");
    }

    return 0;
}
