
# **Estimativa de π com o Método de Monte Carlo em Python**

Este projeto utiliza o **Método de Monte Carlo** para estimar o valor da constante π (pi) por meio de simulações estocásticas implementadas em Python. Além do cálculo, o projeto inclui visualizações gráficas e análise de convergência.

## 📌 Descrição

O algoritmo utiliza a geração de pontos aleatórios dentro de um quadrado unitário para estimar a área de um quarto de círculo inscrito. A razão entre os pontos que caem dentro do círculo e o total de pontos permite uma aproximação de π. O projeto também apresenta:

* Simulação gráfica dos pontos gerados.
* Cálculo do erro absoluto e relativo.
* Gráficos de convergência e erro em escala logarítmica.
* Análise do comportamento do erro em função da quantidade de amostras.

## 🧮 Fórmula

A razão entre os pontos que caem dentro do círculo e o total de pontos tende a $\frac{\pi}{4}$, portanto:

$$
\pi \approx 4 \cdot \frac{\text{Nº de pontos dentro do círculo}}{\text{Nº total de pontos}}
$$

## 📊 Exemplos de Visualizações

* Simulação de pontos dentro e fora do círculo.
* Gráfico da aproximação de π com diferentes quantidades de pontos.
* Gráfico do erro absoluto comparado à tendência $\frac{1}{\sqrt{n}}$.

## 🛠️ Requisitos

* Python 3.7+
* Bibliotecas:

  * `numpy`
  * `matplotlib`
  * `tqdm` (opcional, apenas para exibir barra de progresso)

Você pode instalar os requisitos com:

```bash
pip install numpy matplotlib tqdm
```

## ▶️ Como Executar

Basta rodar o script principal:

```bash
python monte_carlo_pi.py
```

O script:

1. Estima o valor de π com 100.000 pontos.
2. Mostra o erro absoluto e relativo.
3. Exibe a visualização dos pontos.
4. Executa análise de convergência até 1.000.000 de pontos.
5. Plota dois gráficos: aproximação vs. número de pontos, e erro vs. número de pontos.

## 📁 Estrutura

```
monte_carlo_pi.py   # Script principal com simulação e visualizações
README.md           # Este arquivo
```

## 📈 Resultados Esperados

* Aproximação de π com precisão crescente conforme o número de pontos aumenta.
* Gráficos que demonstram a convergência numérica.
* Visualização clara do conceito geométrico por trás do método.

## 📚 Referências

* Kalos, M. H., & Whitlock, P. A. (2008). *Monte Carlo Methods*. Wiley-VCH.
* Press, W. H., et al. (2007). *Numerical Recipes*. Cambridge University Press.
* [https://docs.python.org/](https://docs.python.org/)


