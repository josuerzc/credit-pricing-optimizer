# Credit Pricing Optimizer

Proyecto de portafolio: un modelo de **pricing basado en riesgo**
(risk-based pricing) para préstamos personales, que responde la pregunta
central de todo negocio de crédito retail: *¿qué tasa de interés
ofrecerle a cada cliente para maximizar el resultado del negocio,
sabiendo que una tasa más alta genera más margen por préstamo pero
reduce la probabilidad de que el cliente acepte (y puede empeorar el
mix de riesgo del portafolio por selección adversa)?*

El proyecto combina tres piezas:

1. **Modelo de demanda** — probabilidad de que un cliente acepte un
   préstamo según la tasa ofrecida y su perfil de riesgo.
2. **Optimizador de tasas** — encuentra la tasa óptima por segmento
   usando programación lineal/entera (PuLP).
3. **Frontera eficiente** — 5 escenarios de política de pricing con
   distintos trade-offs entre volumen colocado y utilidad neta.

## Fundamento académico

La función de rentabilidad y el planteamiento de optimización están
basados directamente en:

> Phillips, R. (2013). *Optimizing Prices for Consumer Credit.*
> Columbia Business School, Working Paper Series No. 2013-1.
> http://www.cprm.columbia.edu

De ese paper se toman, sin modificarlas, dos piezas centrales:

**PVNII (Present Value of Net Interest Income)** — Ecuación 4 del
paper, aproximación lineal en la tasa bajo el supuesto de tasa de
descuento interna ≈ 0 y plazos largos:

```
PVNII(P, r, n) ≈ P × [ n×(r - r_c) − PD × LGD ]
```

Es una **resta directa** de la pérdida esperada (marco de Expected Loss
de Basel: EL = PD × LGD × EAD) sobre el margen bruto total del
préstamo a término — **no** lleva un factor `(1-PD)` multiplicando el
margen. Ver la sección de Limitaciones y Metodología en los notebooks
para la justificación completa.

**Problema de optimización** — Ecuación 5 del paper:

```
max_r  TR(r) = Σ_i  D_i × F̄_i(r_i) × [ PVNII(P_i, r_i, n_i) + v_i ]
```

donde `v_i` (ingresos no financieros menos gastos operativos) se omite
en este proyecto siguiendo la misma simplificación que usa Phillips,
quien indica que su efecto es pequeño relativo al ingreso neto de
interés.

## Dataset

Fuente: Kaggle [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
("All Lending Club loan data"), descargado con el Kaggle CLI.

- `accepted_2007_to_2018Q4.csv.gz` — préstamos otorgados (~2.26M filas).
- `rejected_2007_to_2018Q4.csv.gz` — solicitudes rechazadas (~27.6M filas).

**Muestreo:** por el tamaño del dataset completo, se trabaja con una
muestra aleatoria simple reproducible (`random_state=42`) de 150,000
filas de `accepted` y 150,000 de `rejected` (~300k filas en total). El
método (implementado en [`src/download_sample.py`](src/download_sample.py))
cuenta primero el número exacto de filas de datos de cada archivo,
elige exactamente `n_sample` índices sin reemplazo con
`numpy.random.RandomState(42)`, y pasa el resto como `skiprows` a
`pandas.read_csv` — evitando así cargar en memoria las filas
descartadas y garantizando una muestra 100% reproducible sin sesgo de
orden. Las muestras se guardan en `data/raw/` (no versionadas en Git,
ver `.gitignore`; se regeneran corriendo el script).

Para regenerar los datos crudos muestreados:

```bash
python -m src.download_sample
```

## Estructura del repositorio

```
data/
  raw/              muestras descargadas de Kaggle (no versionado)
  processed/        datasets limpios y unidos (no versionado)
notebooks/
  01_eda.ipynb
  02_demand_model.ipynb
  03_risk_estimation.ipynb        PD, LGD individuales, costo de fondeo
  04_aggregation_subgrade.ipynb   agregación individual -> sub_grade
  05_optimization.ipynb
  06_efficient_frontier.ipynb
src/
  download_sample.py  descarga y muestreo reproducible (Kaggle CLI)
  preprocessing.py    limpieza y unión de datasets
  demand_model.py      modelo de demanda a nivel individual
  risk_model.py         modelos individuales de PD y LGD
  aggregation.py        agregación individual -> sub_grade
  optimization.py      optimización con PuLP (Revenue, Profitability, frontera)
requirements.txt
```

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_sample   # descarga + muestreo (requiere Kaggle CLI configurado)
jupyter notebook
```

## Decisión metodológica: granularidad individual vs. sub_grade

Phillips (2013) define "segmento de pricing" de forma flexible: la
Ecuación 5 (la suma sobre `i`) es válida sea que `i` represente una
banda amplia o un cliente individual — es la misma matemática, solo
cambia cuántos términos tiene la suma. Este proyecto usa dos niveles de
granularidad, cada uno con su propósito:

- Los **modelos** (demanda, PD, LGD) se entrenan a nivel de **cliente
  individual**, con sus features continuas reales (FICO, dti, monto,
  etc.). Agregar antes de modelar tiraría a la basura la variación
  individual dentro de cada segmento.
- La **optimización** se resuelve a nivel de **sub_grade** de
  LendingClub (A1...G5, 35 niveles): es el nivel que la propia
  plataforma usa en la vida real para fijar tasas, y computacionalmente
  manejable para PuLP (35 sub_grades × ~15 tasas candidatas ≈ 525
  variables binarias). Optimizar a nivel de ~150,000 clientes
  individuales generaría más de 2M de variables binarias, inviable
  para un solver gratuito una vez que se agregan las restricciones de
  volumen/riesgo de portafolio que conectan a todos los clientes entre
  sí.

El puente entre ambos niveles: para obtener `p_{i,k}`, `PD_i` o `LGD_i`
a nivel de sub_grade `i`, **no** se reentrena un modelo agregado — se
toman todos los clientes reales de ese sub_grade, se corre el modelo
individual sobre cada uno, y se **promedian** sus predicciones. Así el
número que usa el optimizador está informado por toda la heterogeneidad
individual capturada en los modelos. Detalle completo en
[`notebooks/04_aggregation_subgrade.ipynb`](notebooks/04_aggregation_subgrade.ipynb).

## Limitaciones metodológicas reconocidas

- **Proxy de aceptación:** en `rejected` no siempre se distingue si el
  *cliente* rechazó la oferta o el *banco* rechazó al cliente — es un
  proxy razonable de `F̄_i(r)`, no una medición perfecta.
- **PD fija por segmento:** la versión base de este proyecto no
  modela selección adversa (PD dependiente de la tasa ofrecida).
  Phillips mismo trata esto como problema abierto en el paper; queda
  documentado como extensión futura, no implementada en el alcance
  base.
- **Costo de fondeo (`r_c`):** LendingClub es P2P, no banco
  tradicional — el dataset no trae esta variable. Se usa un supuesto
  constante documentado, con análisis de sensibilidad sobre 2-3
  valores.
- **`v_i` omitido:** siguiendo la misma simplificación que usa
  Phillips en el paper original.
- **EAD aproximado:** se usa `out_prncp`/`total_rec_prncp` (saldo
  pendiente al momento del default) en vez del monto original, lo que
  reduce pero no elimina la aproximación.

## Referencias

- Phillips, R. (2013). *Optimizing Prices for Consumer Credit.*
  Columbia Business School, Working Paper 2013-1.
  https://business.columbia.edu/sites/default/files-efs/imce-uploads/CPRM/2013-1_Price_Opt_for_Cons_Credit.pdf
- Basel Committee / Federal Reserve — marco de Pérdida Esperada
  EL = PD × LGD × EAD.
- Federal Reserve Bank of Minneapolis — modelo "cost-plus" de pricing
  bancario: https://www.minneapolisfed.org/article/2000/how-do-lenders-set-interest-rates-on-loans
- "Improving Realized LGD Approximation: A Novel Framework with
  XGBoost for Handling Missing Cash-Flow Data".
  https://arxiv.org/pdf/2406.17308
