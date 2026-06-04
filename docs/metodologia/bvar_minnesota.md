# BVAR com Minnesota prior: metodologia

---

## 1. Motivação

O VAR(4) clássico do Bloco 5 estima 32 parâmetros por equação em 271 observações (6 endógenas × 4 lags = 24 coeficientes autorregressivos, mais 8 exógenas: intercepto + 7 dummies de pandemia). Em amostras dessa magnitude, a estimação por mínimos quadrados ordinários (OLS) produz coeficientes com variância amostral elevada, manifestada empiricamente nos resultados do Bloco 5 em:

- Bandas de confiança amplas para câmbio e commodities (IC 95% englobando intervalos da ordem de ±100% em alguns horizontes)
- Sensibilidade da especificação à seleção de defasagens e ao tratamento de outliers
- Viés de pequena amostra documentado por Kilian (1998), endereçado parcialmente pelo bootstrap-after-bootstrap mas não eliminado

O BVAR (Bayesian Vector Autoregression) endereça essa limitação ao complementar a informação amostral com informação econômica externa, na forma de uma distribuição a priori sobre os coeficientes. O Minnesota prior (Litterman, 1986) é a especificação canônica para essa finalidade em VARs macroeconômicos.

---

## 2. O Minnesota prior

### 2.1 Crenças econômicas codificadas

O Minnesota prior codifica quatro propriedades empíricas tipicamente observadas em séries macroeconômicas:

| Propriedade | Codificação no prior |
|:---|:---|
| Persistência das séries | $\delta_i = 1$ no lag 1 próprio (séries em nível) ou $\delta_i = 0$ (séries em diferença) |
| Cross-lags são fracos | Cross-equation tightness $\theta < 1$ |
| Lags distantes importam menos | Lag decay $\alpha \geq 1$ |
| Variáveis com escalas distintas | Normalização por $\sigma_i / \sigma_j$ |

### 2.2 Calibração adotada neste projeto

| Hiperparâmetro | Valor | Justificativa |
|:---:|:---:|:---|
| $\delta_i$ | 0 para todas as endógenas | As seis endógenas estão em primeira diferença ou são estacionárias por construção (IPCA é taxa mensal de variação, classificada como I(0) pelos testes do Bloco 4) |
| $\theta$ | 0,5 | Convenção de Litterman, padrão na literatura aplicada |
| $\alpha$ | 2 | Decaimento quadrático nos lags, padrão na literatura |
| $\lambda$ | Calibrado | Selecionado via marginal likelihood (Giannone, Lenza & Primiceri, 2015) |

A escolha de $\delta_i = 0$ representa uma adaptação direta da formulação original de Litterman (1986) ao caso de variáveis pré-diferenciadas. Litterman calibrou o prior pensando em séries em nível com forte persistência (PIB, preços, agregados monetários), para as quais o lag 1 da própria variável tem coeficiente próximo de 1. Nas séries em primeira diferença ou em taxas de variação (caso de todas as endógenas deste projeto), o coeficiente esperado é próximo de 0 — codificado no prior como $\delta_i = 0$.

---

## 3. Estrutura matemática

### 3.1 Formulação do VAR em forma matricial

$$Y = XB + U, \quad U_t \sim N(0, \Sigma)$$

onde $Y$ é $(T \times K)$, $X$ é $(T \times Kp + m)$, $B$ é $((Kp+m) \times K)$ e $U$ é $(T \times K)$. No projeto: $K=6$, $p=4$, $m=8$, $T=271$.

### 3.2 Prior conjugado normal-inverse-Wishart

O prior é especificado em forma hierárquica:

$$\Sigma \sim IW(\Psi, d)$$

$$B \mid \Sigma \sim MN(B_0, \Sigma \otimes \Omega_0)$$

Esta é a família de prior **conjugada** com a likelihood Gaussiana do VAR, propriedade que garante posterior em forma fechada.

| Hiperparâmetro do prior | Função |
|:---:|:---|
| $B_0$ | Média a priori dos coeficientes (zeros, conforme calibração da Seção 2.2) |
| $\Omega_0$ | Variância a priori dos coeficientes (diagonal, calibrada pela fórmula de Litterman) |
| $\Psi$ | Escala do prior de $\Sigma$ (diagonal, com $\sigma_i^2$ obtidos de AR(p) univariados preliminares) |
| $d$ | Graus de liberdade do prior de $\Sigma$ ($d = K + 2$, mínimo para a média existir) |

### 3.3 Calibração de $\Omega_0$ — fórmula de Litterman

A variância do prior para o coeficiente do lag $l$ da variável $j$ na equação da variável $i$:

$$\omega_{ij,l}^2 = \begin{cases} \left(\dfrac{\lambda}{l^\alpha}\right)^2 & \text{se } i = j \\[8pt] \left(\dfrac{\lambda \cdot \theta \cdot \sigma_i}{l^\alpha \cdot \sigma_j}\right)^2 & \text{se } i \neq j \end{cases}$$

O termo $\sigma_i / \sigma_j$ normaliza as escalas heterogêneas entre variáveis. Os $\sigma_i$ são obtidos de regressões AR(p) univariadas preliminares.

---

## 4. O hiperparâmetro $\lambda$

$\lambda$ (overall tightness) controla a força global do encolhimento Bayesiano:

| $\lambda$ | Comportamento |
|:---:|:---|
| $\to 0$ | Encolhimento total — BVAR ignora os dados e usa apenas o prior |
| $\to \infty$ | Sem encolhimento — BVAR converge para OLS (= VAR clássico) |
| Intermediário | Combinação ponderada entre prior e informação amostral |

A escolha de $\lambda$ determina a magnitude do trade-off bias-variance:
- Valores baixos: estimadores com menor variância, mas com viés em direção ao prior
- Valores altos: estimadores próximos ao OLS, com maior variância

---

## 5. Posterior em forma fechada

Combinando o prior normal-inverse-Wishart com a likelihood Gaussiana via teorema de Bayes:

$$\Omega_* = (\Omega_0^{-1} + X'X)^{-1}$$

$$B_* = \Omega_*(\Omega_0^{-1} B_0 + X'Y)$$

$$\Psi_* = \Psi + (Y - XB_*)'(Y - XB_*) + (B_* - B_0)'\Omega_0^{-1}(B_* - B_0)$$

$$d_* = d + T$$

A média do posterior $B_*$ é uma média ponderada entre $B_0$ (prior) e $\hat{B}_{OLS} = (X'X)^{-1}X'Y$ (estimador OLS), com pesos dados pela precisão do prior ($\Omega_0^{-1}$) e pela informação amostral ($X'X$).

**Casos-limite:**

- Prior fraco ($\Omega_0 \to \infty$): $\Omega_0^{-1} \to 0$, portanto $B_* \to \hat{B}_{OLS}$
- Prior forte ($\Omega_0 \to 0$): $\Omega_0^{-1} \to \infty$, portanto $B_* \to B_0$

Esta estrutura é a base matemática do encolhimento Bayesiano: coeficientes estimados são puxados em direção ao ponto-âncora do prior, com a magnitude do efeito controlada pela razão entre precisão do prior e informação amostral.

---

## 6. Calibração de $\lambda$ via marginal likelihood

### 6.1 Justificativa metodológica

Em vez de fixar $\lambda$ por convenção (abordagem de Litterman, 1986) ou por validação cruzada (computacionalmente custosa), este projeto adota a abordagem de Giannone, Lenza & Primiceri (2015): selecionar $\lambda$ pela maximização da **marginal likelihood**, definida como:

$$p(Y \mid \lambda) = \int p(Y \mid B, \Sigma) \cdot p(B, \Sigma \mid \lambda) \, dB \, d\Sigma$$

A marginal likelihood opera simultaneamente como:

1. Medida de ajuste aos dados observados
2. Penalização automática de complexidade, via integração sobre o espaço de parâmetros (modelos com prior fraco ocupam maior "volume" no espaço, recebendo penalização implícita)
3. Procedimento que dispensa validação cruzada — toda a validação está implícita na integração

### 6.2 Forma fechada

Para o prior conjugado normal-inverse-Wishart, a marginal likelihood possui forma analítica fechada, derivável a partir das propriedades da distribuição matrix-normal e da gamma multivariada. Não há necessidade de MCMC ou integração numérica.

### 6.3 Procedimento operacional

1. Definir grade de candidatos para $\lambda$ em escala logarítmica (intervalo típico: 0,01 a 5)
2. Para cada candidato, computar os hiperparâmetros do posterior em forma fechada
3. Avaliar a log marginal likelihood
4. Selecionar o $\lambda$ que maximiza a função

---

## 7. Construção das IRFs Bayesianas

### 7.1 Diferença conceitual em relação ao VAR clássico

No VAR clássico, a IRF é uma função determinística dos coeficientes estimados. No BVAR, a IRF herda a distribuição posterior dos parâmetros — torna-se, portanto, uma variável aleatória com distribuição posterior completa.

### 7.2 Procedimento de amostragem

1. Amostre $\Sigma^{(s)} \sim IW(\Psi_*, d_*)$
2. Amostre $B^{(s)} \sim MN(B_*, \Sigma^{(s)} \otimes \Omega_*)$
3. Compute a decomposição de Cholesky de $\Sigma^{(s)}$ para identificação ortogonal
4. Itere o sistema com $B^{(s)}$ para obter a IRF do passo $s$
5. Repita $S$ vezes (típico: $S = 5000$)
6. Bandas de credibilidade são os quantis empíricos da distribuição amostrada

### 7.3 Contraste com o bootstrap Kilian (Bloco 5)

| Aspecto | Bootstrap Kilian | BVAR |
|:---|:---:|:---:|
| Filosofia | Frequentista | Bayesiana |
| O que é aleatorizado | Dados artificiais | Parâmetros do modelo |
| Mecanismo | Reamostragem de resíduos + re-estimação | Amostragem direta do posterior |
| Bandas representam | Variabilidade amostral | Incerteza posterior |
| Tratamento de pequena amostra | Correção de viés do OLS | Encolhimento via prior |

Ambas as abordagens produzem distribuições para IRFs, mas suas interpretações filosóficas e resultados quantitativos podem divergir, especialmente quando o prior carrega informação substantiva.

---

## 8. Hipóteses empíricas a testar

Os resultados do Bloco 5 estabeleceram o seguinte quadro:

- Transmissão para atividade econômica (IBC-Br): significativa a 95% nos horizontes 12-24 meses
- Transmissão para preços (IPCA): inconclusiva a 95%, com price puzzle marginal a 68% nos primeiros 6 meses
- Canal cambial: sinal econômico ambíguo e estatisticamente nulo

O BVAR é avaliado neste bloco quanto à sua capacidade de:

1. **Redução de variância amostral**: a estabilização via prior pode estreitar bandas suficientemente para revelar transmissão monetária para preços antes inconclusiva
2. **Robustez em pequena amostra**: o BVAR é menos sensível a especificação e a outliers que o OLS, propriedade desejável dada a heterogeneidade de regimes na janela 2003-2025
3. **Comparabilidade metodológica**: replicação do exercício metodológico que falhou na monografia original (Delmondes, 2024)

---

## 9. Implementação

O Bloco 6 está organizado em três fases:

| Fase | Conteúdo | Notebook |
|:---:|:---|:---|
| 6.1 | Implementação manual com prior conjugado em forma fechada | `06_bvar.ipynb`, seções 1-5 |
| 6.2 | Replicação via PyMC (framework Bayesiano padrão em Python) | `06_bvar.ipynb`, seções 6-7 |
| 6.3 | Comparação estruturada com VAR clássico do Bloco 5 | `06_bvar.ipynb`, seção 8 |

A Fase 6.1 implementa o BVAR do zero, com transparência total sobre as derivações Bayesianas. A Fase 6.2 reimplementa o mesmo modelo via PyMC, com fins de demonstrar a abordagem com pacote consolidado. A Fase 6.3 estabelece a comparação rigorosa entre VAR clássico (Bloco 5) e BVAR (Bloco 6) em termos de IRFs, FEVD e veredito sobre a transmissão monetária para preços.

---

## Referências

Bańbura, M., Giannone, D. & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*, 25(1), 71–92.

Giannone, D., Lenza, M. & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *Review of Economics and Statistics*, 97(2), 436–451.

Kadiyala, K. R. & Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR-models. *Journal of Applied Econometrics*, 12(2), 99–132.

Karlsson, S. (2013). Forecasting with Bayesian Vector Autoregression. In G. Elliott & A. Timmermann (Eds.), *Handbook of Economic Forecasting*, Vol. 2B, Ch. 15. Elsevier.

Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics*, 80(2), 218–230.

Koop, G. & Korobilis, D. (2010). Bayesian multivariate time series methods for empirical macroeconomics. *Foundations and Trends in Econometrics*, 3(4), 267–358.

Lenza, M. & Primiceri, G. E. (2022). How to estimate a vector autoregression after March 2020. *Journal of Applied Econometrics*, 37(4), 688–699.

Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions — Five years of experience. *Journal of Business & Economic Statistics*, 4(1), 25–38.