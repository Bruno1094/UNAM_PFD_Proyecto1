import streamlit as st

from motor.vanilla import europea
from paginas.comun import K_T, caja, encabezado, mercado


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Derivados en tiempo discreto",
               "Calculadora de derivados. Los datos de la izquierda alimentan forwards, Black-Scholes y arbitraje; "
               "los árboles tienen sus propios datos y ejemplos de clase.")
    c = st.columns(4)
    caja("Precio forward F", float(m.forward(T)), cont=c[0])
    caja("Call europeo", float(europea(m, K, T, "call")), cont=c[1])
    caja("Put europeo", float(europea(m, K, T, "put")), cont=c[2])
    caja("B(0,T)", float(m.B(T)), secundaria=True, cont=c[3])

    st.header("Índice")
    izq, der = st.columns(2)
    izq.markdown("""

**Forwards.** Valor, precio y paridad put-call, con dividendos e ingreso fijo.

**Árboles.** Un periodo, N periodos, réplica en una trayectoria, fórmula binomial y árbol no recombinante.

**Martingalas.** Filtración, martingalas, teorema de representación y Radon-Nikodym.

**Black-Scholes.** Límite del árbol, fórmula y otros pagos X(S(T)).
""")
    der.markdown("""
**Americanas.** Ejercicio óptimo, consumo, tiempos de paro y muestreo opcional.

**Arbitraje.** Paridad, forward, teórico contra mercado, árbol y cadena de strikes, con la estrategia completa.

**Calculadoras.** Opciones, exóticos y un constructor de contratos a la medida.

**Complementos.** Griegas. Estrategias y riesgo: Arma tu portafolio.
""")
