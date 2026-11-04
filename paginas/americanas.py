import pandas as pd
import streamlit as st

from motor import curso as cu
from paginas import arbol_ui
from paginas.comun import aviso, caja, encabezado, graficar, tabla


def pagina():
    encabezado("Opciones americanas", "Valúa opciones que pueden ejercerse en cualquier momento, comparando en cada nodo el valor de ejercerlas con el de conservarlas.")
    st.caption("X(nδ) es el valor intrínseco y V(nδ) el valor del derivado.")
    d = arbol_ui.definir("Árbol en uso", forzar="americana", inicial="Sesión 8: put americano K = 5")
    if not d:
        return
    a, v = d["a"], d["v"]
    eu = cu.valuar(a, d["expr"], "europea", **d["ctes"])
    if a.N > 6:
        aviso("Reduce el número de periodos", "Estas tablas se leen bien hasta con 6 periodos.", "neutro")
        return
    t1, t2, t3 = st.tabs(["Valuación y consumo", "Tiempos de paro", "Muestreo opcional"])

    with t1:
        c = st.columns(3)
        caja("Valor americano h₀", v.precio, cont=c[0])
        caja("Valor europeo", eu.precio, secundaria=True, cont=c[1])
        caja("Prima por ejercer antes", v.precio - eu.precio, secundaria=True, cont=c[2])
        st.latex(r"h_N(y)=X(y),\qquad h_k(y)=\max\{B(k\delta,(k+1)\delta)[q(k\delta)h_{k+1}(uy)+(1-q(k\delta))h_{k+1}(dy)],\ X(y)\}")
        if a.N <= 5:
            graficar(arbol_ui.dibujar(v))
            st.caption("Los rombos rojos son nodos donde conviene ejercer.")
        filas = []
        for j in range(1, 2 ** a.N):
            filas.append({"Nodo": j, "ω": cu.nombre_nodo(j), "S": a.S[j], "X intrínseco": v.X[j], "Dejar vivir": v.cont[j], "V = h_k": v.V[j],
                          "Conviene": "Ejercer" if v.ejercer[j] else "Continuar", "Consumo si no ejerce": v.V[j] - v.cont[j], "Δ": v.alfa[j]})
        tabla(pd.DataFrame(filas))
        st.markdown("**Consumo.** Si conviene ejercer y el poseedor no lo hace, el valor cae de X a dejar vivir. "
                    "El emisor puede consumir esa brecha y seguir con la cobertura habitual, calculada con el valor de dejar vivir.")
        st.latex(r"\Delta(k\delta)=\frac{h_{k+1}(uS(k\delta))-h_{k+1}(dS(k\delta))}{(u-d)S(k\delta)}")

    with t2:
        tau = cu.tiempo_optimo(v)
        rho = cu.tiempo_minimo(a)
        ok_t, _ = cu.es_tiempo_de_paro(tau)
        ok_r, viol = cu.es_tiempo_de_paro(rho)
        st.markdown("Un **tiempo de paro** cumple que $\\{\\tau=k\\}\\in F(k)$: para decidir si se para en k basta la información hasta k.")
        c = st.columns(2)
        caja("τ = primer k con h_k = X", "Sí es tiempo de paro" if ok_t else "No es tiempo de paro", "regla óptima de ejercicio", True, cont=c[0])
        caja("ρ = fecha del mínimo de S", "Sí es tiempo de paro" if ok_r else "No es tiempo de paro", "se necesita conocer el futuro" if not ok_r else "", True, cont=c[1])
        filas = [{"ω": "0" + w, "τ(ω)": tau[w], "ρ(ω)": rho[w]} for w in cu.todas_trayectorias(a.N)]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        if viol:
            k, w, otro = viol[0]
            st.caption(f"Ejemplo: 0{w} y 0{otro} coinciden hasta {k}δ, pero ρ para en {k} solo en la primera, así que {{ρ = {k}}} no está en F({k}).")

    with t3:
        tau = cu.tiempo_optimo(v)
        st.latex(r"Z\text{ martingala},\ \tau\le\rho\ \Rightarrow\ Z(\tau)=E[Z(\rho)\mid F(\tau)]")
        st.markdown("Con ρ = N y Z = B·S, en cada átomo de F(τ):")
        st.caption("Si Z es submartingala vale ≤; si es supermartingala, ≥.")
        tabla(pd.DataFrame(cu.muestreo_opcional(v, tau)))
