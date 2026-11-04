import numpy as np
import pandas as pd
import streamlit as st

from motor import curso as cu
from paginas import arbol_ui
from paginas.comun import aviso, caja, encabezado, lista_num, tabla, txt


def pagina():
    encabezado("Martingalas y cambio de medida", "Sesiones 5 y 6: por qué el precio de la cobertura es una esperanza bajo Q.")
    d = arbol_ui.definir("Árbol en uso", forzar="europea", inicial="Sesión 5: árbol de la martingala")
    if not d:
        return
    a, v = d["a"], d["v"]
    if a.N > 6:
        aviso("Reduce el número de periodos", "Estas tablas se leen bien hasta con 6 periodos.", "neutro")
        return
    t1, t2, t3, t4 = st.tabs(["Filtración", "Martingalas", "Teorema de representación", "Radon-Nikodym"])

    with t1:
        st.markdown("Cada σ-álgebra describe la historia del precio hasta nδ. Aquí se identifica con las trayectorias de nodos.")
        filas = []
        for n in range(a.N + 1):
            conj = cu.filtracion(a.N, n)
            filas.append({"Fecha": f"{n}δ", "F(n)": "{ " + ", ".join("{" + ",".join(map(str, c)) + "}" for c in conj[:16]) + (" ..." if len(conj) > 16 else "") + " }"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.latex(r"F(n)\subset F(n+1),\qquad F(0)=\{\varnothing,\Omega\},\qquad E_P[X\mid F(0)]=E_P[X],\qquad E_P[X\mid F(N)]=X")
        st.markdown("**Proceso previsible:** α(nδ) ya se conoce en (n-1)δ. Por eso la cartera se puede armar antes de ver el siguiente precio.")

    with t2:
        p_txt = txt("Probabilidad real p (una, o una por nodo no final)", "w_m_p", "0.5")
        pv = lista_num(p_txt)
        try:
            if not pv:
                raise ValueError("Escribe p como número, por ejemplo 0.5.")
            p = cu.probabilidad_p(a, pv)
        except ValueError as e:
            aviso("Revisa p", str(e))
            return
        finales = cu.nivel(a.N)
        Y_P = cu.esperanza_condicional(a, a.S[finales], p)
        filas = []
        Q_S = cu.esperanza_condicional(a, a.S[finales], v.q)
        for j in range(1, 2 ** (a.N + 1)):
            filas.append({"Nodo": j, "ω": cu.nombre_nodo(j), "S(nδ)": a.S[j], "E_P[S(Nδ) | F(n)]": Y_P[j], "E_Q[S(Nδ) | F(n)]": Q_S[j]})
        tabla(pd.DataFrame(filas))
        Z = cu.descuento_acumulado(a) * a.S
        Z[0] = np.nan
        pruebas = [("S bajo P", cu.es_martingala(a, a.S, p)[0]), ("E_P[S(Nδ)|F(n)] bajo P", cu.es_martingala(a, Y_P, p)[0]),
                   ("S bajo Q", cu.es_martingala(a, a.S, v.q)[0]), ("Z = B·S bajo Q", cu.es_martingala(a, Z, v.q)[0])]
        c = st.columns(4)
        for i, (nom, ok) in enumerate(pruebas):
            caja(nom, "Sí es martingala" if ok else "No es martingala", secundaria=True, cont=c[i])
        st.latex(r"E_P[Y(i\delta)\mid F(j)]=Y(j\delta),\ \ i\ge j\qquad\text{Torre: }E[E[Y\mid F(i)]\mid F(j)]=E[Y\mid F(j)],\ \ j\le i")
        st.caption("El proceso E_P[X|F(n)] siempre es P-martingala, por la propiedad de torre.")

    with t3:
        Z, Y, alfa = cu.representacion(v)
        st.latex(r"Z(n\delta)=B(0,n\delta)S(n\delta),\quad Y(n\delta)=E_Q[B(0,N\delta)X\mid F(n)],\quad Y(n\delta)=Y(0)+\sum_{k=1}^n\alpha(k\delta)\Delta Z(k\delta)")
        c = st.columns(2)
        caja("Y(0) = valor del derivado", Y[1], "π(0,0) = B(0,Nδ) E_Q[X]", cont=c[0])
        coincide = bool(np.allclose(alfa[1:], v.alfa[1:2 ** a.N], equal_nan=True))
        caja("¿α del teorema = α de la cobertura?", "Sí" if coincide else "No", secundaria=True, cont=c[1])
        filas = []
        for j in range(1, 2 ** (a.N + 1)):
            f = {"Nodo": j, "ω": cu.nombre_nodo(j), "Z": Z[j], "Y": Y[j]}
            if j < 2 ** a.N:
                f["α = δY/δZ"] = alfa[j]
                f["α cobertura"] = v.alfa[j]
            filas.append(f)
        tabla(pd.DataFrame(filas))
        st.markdown("**Valor en nδ:** $B^{-1}(0,n\\delta)\\,E_Q[B(0,N\\delta)X\\mid F(n)]$, que con B determinista es $B(n\\delta,N\\delta)E_Q[X\\mid F(n)]$.")
        st.markdown("La estrategia guarda $\\alpha((n+1)\\delta)$ unidades de subyacente y $\\beta((n+1)\\delta)=Y(n\\delta)-\\alpha((n+1)\\delta)Z(n\\delta)$ en la cuenta en efectivo.")

    with t4:
        r = cu.radon_nikodym(v, p)
        finales = cu.nivel(a.N)
        X = v.V[finales]
        if not r["equivalentes"]:
            aviso("P y Q no son equivalentes", "No concuerdan en lo que es posible, así que dQ/dP no existe en todas las trayectorias.")
        filas = []
        for i, j in enumerate(finales, start=1):
            filas.append({"i": i, "Trayectoria": cu.nombre_nodo(int(j)), "τ(i)": r["tau"][j], "τ̂(i)": r["tau_q"][j],
                          "dQ/dP(i)": r["zeta"][j], "X(i)": X[i - 1]})
        tabla(pd.DataFrame(filas))
        if r["equivalentes"]:
            Ep = float(np.sum(r["tau"][finales] * X))
            Eq = float(np.sum(r["tau_q"][finales] * X))
            Er = float(np.sum(r["tau"][finales] * r["zeta"][finales] * X))
            c = st.columns(3)
            caja("E_P[X]", Ep, secundaria=True, cont=c[0])
            caja("E_Q[X]", Eq, cont=c[1])
            caja("E_P[(dQ/dP) X]", Er, "debe igualar a E_Q[X]", True, cont=c[2])
        st.latex(r"\frac{dQ}{dP}(i):=\frac{\hat\tau(i)}{\tau(i)},\qquad E_Q[X]=\sum_i\hat\tau(i)X(i)=E_P\left[\frac{dQ}{dP}X\right],\qquad\zeta(j\delta)=E_P\left[\frac{dQ}{dP}\Bigm|F(j)\right]")
        filas = [{"Nodo": j, "ω": cu.nombre_nodo(j), "ζ(nδ)": r["zeta"][j]} for j in range(1, 2 ** (a.N + 1))]
        with st.expander("Proceso ζ en todos los nodos"):
            tabla(pd.DataFrame(filas))
        st.latex(r"E_Q[X(i\delta)\mid F(j)]=\zeta^{-1}(j\delta)\,E_P[\zeta(i\delta)X(i\delta)\mid F(j)]")
        st.caption("Dos medidas son equivalentes si P(A) > 0 ⇔ Q(A) > 0: concuerdan en lo que es posible.")
