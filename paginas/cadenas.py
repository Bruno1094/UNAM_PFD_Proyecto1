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
    encabezado("Arbitraje", "Detecta oportunidades de ganancia sin riesgo cuando dos instrumentos con el mismo pago final tienen precios distintos hoy.")
    t1, t2, t3, t4, t5 = st.tabs(["Paridad put-call", "Forward", "Teórico contra mercado", "Árbol", "Cadena de strikes"])
        
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
            aviso("No se encontró arbitraje", "Si las revisiones de precios se cumplen, no hay oportunidades de arbitraje que aprovechar.", "neutro")
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
