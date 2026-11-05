# Derivados en tiempo discreto

Calculadora web para valuar derivados, replicarlos y detectar oportunidades de arbitraje. Todo funciona capturando datos, sin escribir código.

**[Abrir la aplicación](https://unampfdproyecto1-ssywkldwci2appvclvj596t.streamlit.app)**

## Idea central

El precio de un derivado es el costo de replicarlo. Si dos portafolios valen lo mismo al vencimiento en todo escenario, deben valer lo mismo hoy. Si no, existe arbitraje.

## Qué incluye

| Pestaña | Contenido |
|---|---|
| **Forwards** | Precio, valor, payoffs y paridad put-call, con dividendos o ingreso fijo. |
| **Árboles** | Modelo binomial de uno y N periodos, cobertura (α, β) en cada nodo, autofinanciamiento y árbol no recombinante. |
| **Martingalas** | Filtración, medida de riesgo neutral, teorema de representación y derivada de Radon-Nikodym. |
| **Black-Scholes** | Límite del árbol, fórmula cerrada y valuación de otros pagos X(S(T)). |
| **Americanas** | Ejercicio óptimo, consumo, tiempos de paro y muestreo opcional. |
| **Arbitraje** | Paridad put-call, forwards, precio teórico contra mercado y árboles, con la estrategia y sus flujos. |
| **Calculadoras** | Opciones europeas, americanas, Bermuda y asiáticas; exóticos, y un constructor de contratos a la medida. |
| **Complementos** | Cadenas de strikes, estrategias con opciones, escenarios y VaR. |

## Cómo se usa

1. Captura los datos de mercado en la barra lateral: S₀, K, T, σ, tasa o B(0,T), e ingresos del subyacente.
2. Elige una pestaña. Los árboles tienen sus propios datos y ejemplos resueltos.
3. Revisa el resultado y cómo se obtuvo.

## Métodos

- **Árboles binomiales:** valuación hacia atrás, incluidos pagos que dependen de la trayectoria.
- **Fórmulas cerradas e integración numérica:** Black-Scholes y exóticos que las admiten.
- **Monte Carlo:** exóticos y contratos a la medida, con semilla fija para repetir resultados.
- **Longstaff-Schwartz:** ejercicio anticipado en el constructor. Da una cota inferior del precio.

Los resultados dependen de los supuestos del modelo, como la volatilidad constante. Esta aplicación es una herramienta de cálculo y estudio, no una recomendación de inversión.

## Ejecutarla en tu computadora

Requiere Python 3.12 o más reciente.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```text
app.py            Archivo principal: barra lateral y navegación
paginas/          Interfaz, un archivo por pestaña
motor/            Cálculos, sin interfaz
.streamlit/       Colores y tipografía
requirements.txt  Bibliotecas necesarias
```

## Autores

- Hernández Alcántara Cristina Geraldine
- Mendoza Aragón Rogelio
- Moreno Ventura Miguel Angel
- Olavarria Guerra Bruno
- Reyes Díaz Axel Gabriel
- Salgado Bahena Arturo
