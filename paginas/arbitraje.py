import numpy as np
import pandas as pd
import streamlit as st

from motor import arbitraje as ar
from motor import vanilla as va
from paginas import arbol_ui
from paginas.comun import (K_T, aviso, caja, casilla, encabezado, figura, graficar, lineas, mercado, mostrar_estrategia,
                           num, radio, reemplazar_tabla, sel, tabla, tabla_editable)


def _cotizaciones(prefijo, teo_c, teo_p, m):
    """Campos de precio. Con bid/ask activado se pide cada lado."""
    detalle = casilla("Capturar bid y ask, y comisiones", f"w_{prefijo}_ba")
    if detalle:
        c = st.columns(3)
        cb = num("Call bid", f"w_{prefijo}_cb", round(teo_c, 2), c[0], min_value=0.0)
        ca = num("Call ask", f"w_{prefijo}_ca", round(teo_c, 2) + .1, c[0], min_value=0.0)
        pb = num("Put bid", f"w_{prefijo}_pb", round(teo_p, 2), c[1], min_value=0.0)
        pa = num("Put ask", f"w_{prefijo}_pa", round(teo_p, 2) + .1, c[1], min_value=0.0)
        sb = num("Subyacente bid", f"w_{prefijo}_sb", float(m.S0), c[2], min_value=0.0)
        sa = num("Subyacente ask", f"w_{prefijo}_sa", float(m.S0), c[2], min_value=0.0)
        co = num("Comisión por operación", f"w_{prefijo}_co", 0.0, min_value=0.0, step=.01)
        return cb, ca, pb, pa, sb, sa, co
    c = st.columns(2)
    cm = num("Precio del call", f"w_{prefijo}_c", round(teo_c, 2) + .5, c[0], min_value=0.0)
    pm = num("Precio del put", f"w_{prefijo}_p", round(teo_p, 2), c[1], min_value=0.0)
    return cm, cm, pm, pm, m.S0, m.S0, 0.0


def _cadena_ejemplo(m, K, T):
    ks = [round(K * x, 2) for x in (.8, .9, 1, 1.1, 1.2)]
    return pd.DataFrame({"K": ks, "Call": [round(float(va.europea(m, k, T)), 3) for k in ks],
                         "Put": [round(float(va.europea(m, k, T, "put")), 3) for k in ks]})


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Arbitraje", "Si dos cosas valen lo mismo en T en todo escenario, deben valer lo mismo hoy. Captura precios de mercado y, si no se cumple, aquí tienes la estrategia.")
    t1, t2, t3, t4, t5 = st.tabs(["Paridad put-call", "Forward", "Teórico contra mercado", "Árbol", "Cadena de strikes"])

    with t1:
        estilo = radio("Estilo de las opciones", "w_ap_e", ["Europeas", "Americanas"])
        teo_c, teo_p = float(va.europea(m, K, T)), float(va.europea(m, K, T, "put"))
        q = _cotizaciones("ap", teo_c, teo_p, m)
        if estilo == "Europeas":
            st.latex(r"c-p=S_0B_E(0,T)-C-KB(0,T)=" + f"{float(m.valor_forward(K, T)):,.4f}")
            e = ar.paridad_europea(m, K, T, q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        else:
            st.latex(r"S_0B_E(0,T)-C-K\;\le\;c-p\;\le\;S_0-KB(0,T)")
            e = ar.paridad_americana(m, K, T, q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        mostrar_estrategia(e)

    with t2:
        F = float(m.forward(T))
        detalle = casilla("Capturar bid, ask y comisiones", "w_af_ba")
        if detalle:
            c = st.columns(5)
            fb = num("F bid", "w_af_fb", round(F, 2), c[0], min_value=0.0)
            fa = num("F ask", "w_af_fa", round(F, 2) + .1, c[1], min_value=0.0)
            sb = num("S bid", "w_af_sb", float(m.S0), c[2], min_value=0.0)
            sa = num("S ask", "w_af_sa", float(m.S0), c[3], min_value=0.0)
            co = num("Comisión", "w_af_co", 0.0, c[4], min_value=0.0)
        else:
            fb = fa = num("Precio forward de mercado", "w_af_f", round(F, 2) + 1.0, min_value=0.0)
            sb = sa = m.S0
            co = 0.0
        caja("F teórico", F, "con S₀, tasas e ingresos de la barra lateral", True)
        mostrar_estrategia(ar.forward_mercado(m, T, fb, fa, sb, sa, co))

    with t3:
        c = st.columns(3)
        tipo = radio("Tipo", "w_am_t", ["Call", "Put"], cont=c[0]).lower()
        est = radio("Estilo", "w_am_e", ["Europea", "Americana"], cont=c[1]).lower()
        pasos = 400
        teo = va.precio_estilo(m, K, T, tipo, est, pasos)
        pm = num("Precio de mercado", "w_am_p", round(teo, 2) + .75, c[2], min_value=0.0)
        inf, sup = ar.cotas(m, K, T, tipo, est)
        c = st.columns(3)
        caja("Precio teórico", teo, f"con σ = {m.sigma:.2%}", cont=c[0])
        caja("Cotas libres de modelo", f"{inf:,.3f} a {sup:,.3f}", secundaria=True, cont=c[1])
        iv = va.vol_implicita(pm, m, K, T, tipo, est, 200)
        caja("Volatilidad implícita del mercado", f"{iv:.2%}" if iv else "no existe", secundaria=True, cont=c[2])
        e = ar.revisar_cotas(m, K, T, tipo, est, pm)
        if e.existe:
            mostrar_estrategia(e)
        else:
            valuar = lambda mm, tt: va.precio_estilo(mm, K, tt, tipo, est, 200)
            delta = va.griegas(valuar, m, T)["Delta"]
            mostrar_estrategia(ar.contra_modelo(m, K, T, tipo, pm, teo, delta))

    with t4:
        st.caption("En el árbol el mercado es completo: cualquier precio distinto al de la cobertura es arbitraje puro.")
        tipo_arb = radio("Árbol", "w_ab_tipo", ["Un periodo", "Varios periodos"])
        if tipo_arb == "Un periodo":
            c = st.columns(3)
            S0 = num("S₀", "w_ab_S", 100.0, c[0], min_value=.001)
            Su = num("S arriba", "w_ab_Su", 120.0, c[1], min_value=.001)
            Sd = num("S abajo", "w_ab_Sd", 90.0, c[2], min_value=.001)
            c = st.columns(4)
            Xu = num("Pago arriba X_u", "w_ab_Xu", 20.0, c[0])
            Xd = num("Pago abajo X_d", "w_ab_Xd", 0.0, c[1])
            B = num("B(0,T)", "w_ab_B", .95, c[2], min_value=.0001, step=.01)
            V = num("Precio de mercado del derivado", "w_ab_V", 10.0, c[3])
            if Su <= Sd:
                aviso("Revisa los escenarios", "S arriba debe ser mayor que S abajo.")
            else:
                st.latex(r"\text{No hay arbitraje}\iff S_d<S_0B^{-1}(0,T)<S_u\iff 0<q<1")
                r = va.bernoulli(S0, Su, Sd, Xu, Xd, B)
                if 0 < r["q"] < 1:
                    caja("Valor de la cobertura", r["V0"], f"α = {r['alfa']:,.4f}, β = {r['beta']:,.4f}", True)
                mostrar_estrategia(ar.binomial_mercado(S0, Su, Sd, B, Xu, Xd, V))
        else:
            d = arbol_ui.definir("Árbol en uso")
            if d is not None:
                v = d["v"]
                c = st.columns(2)
                caja("Valor del árbol", v.precio, cont=c[0])
                V = num("Precio de mercado del derivado", "w_ab_Vn", round(v.precio, 2) + 1.0, c[1])
                if v.estilo == "americana":
                    aviso("Usa un derivado europeo", "La estrategia dinámica de esta sección es para derivados europeos.", "modelo")
                else:
                    mostrar_estrategia(ar.arbitraje_arbol(v, V))

    with t5:
        st.caption("Captura precios por strike para el mismo vencimiento. Se revisan paridad, monotonía, pendiente, mariposas y box.")
        c = st.columns([1, 1, 1.4])
        minimo = num("Ganancia mínima a reportar", "w_ac_min", .01, c[2], min_value=0.0, step=.01,
                     help="Ignora diferencias menores, que suelen venir del redondeo de los precios o de los costos de operar.")
        if c[0].button("Cargar precios teóricos"):
            reemplazar_tabla("cadena", _cadena_ejemplo(m, K, T))
        if c[1].button("Agregar error de ejemplo"):
            df = _cadena_ejemplo(m, K, T)
            df.loc[2, "Call"] = round(df.loc[2, "Call"] + 2.5, 3)
            reemplazar_tabla("cadena", df)
        df = tabla_editable("cadena", _cadena_ejemplo(m, K, T), num_rows="dynamic", hide_index=True, width="stretch")
        df = df.dropna(subset=["K"])
        df = df[df["K"] > 0].drop_duplicates("K")
        if len(df) < 2:
            aviso("Faltan datos", "Captura al menos dos strikes.")
            return
        hall = ar.cadena(m, T, df["K"].to_numpy(float), df["Call"].to_numpy(float), df["Put"].to_numpy(float), minimo)
        if not hall:
            aviso("No encontré arbitraje", "La cadena cumple todas las revisiones.", "neutro")
        else:
            aviso(f"Encontré {len(hall)} señales de arbitraje", "Están ordenadas por ganancia. Un solo precio malo suele aparecer en varias revisiones.")
            for i, e in enumerate(hall[:12]):
                with st.expander(f"{e.titulo}: ganancia hoy {e.ganancia_hoy:,.4f}", expanded=i == 0):
                    tabla(pd.DataFrame(e.piernas))
                    st.caption(e.nota)

        ord_ = df.sort_values("K")
        iv_c = [va.vol_implicita(float(p), m, float(k), T, "call") if not np.isnan(p) else None for k, p in zip(ord_["K"], ord_["Call"])]
        iv_p = [va.vol_implicita(float(p), m, float(k), T, "put") if not np.isnan(p) else None for k, p in zip(ord_["K"], ord_["Put"])]
        fig = figura("Sonrisa de volatilidad implícita", "Strike", "Volatilidad implícita", 300)
        fig.add_scatter(x=ord_["K"], y=iv_c, name="Calls", mode="lines+markers", line=dict(color="#2a62d6", width=2), marker=dict(size=9))
        fig.add_scatter(x=ord_["K"], y=iv_p, name="Puts", mode="lines+markers", line=dict(color="#d1344b", width=2, dash="dash"), marker=dict(size=9, symbol="diamond"))
        fig.update_yaxes(tickformat=".3%")
        graficar(fig)
