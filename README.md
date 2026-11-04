# Derivados en tiempo discreto

Calculadora web en Streamlit basada en las notas del curso: forwards, paridad put-call, árboles binomiales, martingalas, Radon-Nikodym, Black-Scholes como límite, opciones americanas, exóticos y arbitraje.

## Abrirla en tu computadora

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicarla

Sigue `GUIA_PUBLICAR.md`.

## Estructura

- `app.py`: barra lateral y navegación. Es el archivo principal.
- `paginas/`: una pestaña por archivo.
- `motor/`: los cálculos, sin interfaz. `motor/curso.py` tiene el árbol de clase.
- `pruebas/`: pruebas automáticas con los ejercicios de las sesiones. Se corren con `python -m pytest pruebas -q`.
- `.streamlit/config.toml`: colores y tipografía.

## Convenciones

- Nodos: del nodo j salen 2j (abajo) y 2j+1 (arriba). Trayectorias: ω = 0uud.
- Árboles con σ y r: μ = r - ½σ², la elección del curso.
- Asiáticas en árbol: el promedio incluye S(0), como en la sesión 8.
- Ingreso fijo: la volatilidad se aplica a S₀ - C.
- Barreras y lookback con fórmula: monitoreo continuo.
- Monte Carlo usa semilla fija, así que los resultados se pueden repetir.
