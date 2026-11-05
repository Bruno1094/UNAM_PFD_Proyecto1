"""Calculadora de derivados.  Ejecutar con:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import config as _config

from motor.base import Mercado
from paginas import (americanas, arboles, arbitraje, black_scholes, conceptos, constructor, estrategias, exoticos, forwards,
                     griegas, inicio, martingalas, opciones)
from paginas.comun import CSS, asentar_tablas, num, sel, tabla_editable

# La paleta solo está pensada para fondo claro, aunque se pierda la carpeta .streamlit al subir el proyecto.
_config.set_option("theme.base", "light")

st.set_page_config(page_title="Derivados en tiempo discreto", page_icon="∂", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)

# Conserva los valores capturados aunque el campo no se muestre en la pestaña actual.
for k in [k for k in st.session_state if str(k).startswith("w_")]:
    st.session_state[k] = st.session_state[k]


def restablecer():
    s = st.session_state
    s["w_S0"], s["w_K"], s["w_r"], s["w_sigma"], s["w_q"], s["w_T"] = 100.0, 100.0, 5.0, 20.0, 2.0, 1.0
    s["w_ingreso"], s["w_conv"] = "Sin ingresos", "Tasa continua"


with st.sidebar:
    st.header("Datos de Mercado")
    st.caption("Alimenta las siguientes pestañas:\n"
               "Forwards, Black-Scholes, Arbitraje y Calculadoras.")
    c1, c2 = st.columns(2)
    S0 = num("Subyacente S₀", "w_S0", 100.0, c1, min_value=0.0001, step=1.0)
    K = num("Strike K", "w_K", 100.0, c2, min_value=0.0001, step=1.0)
    c1, c2 = st.columns(2)
    T = num("Vencimiento T (años)", "w_T", 1.0, c1, min_value=0.0028, step=0.25)
    sigma = num("Volatilidad σ (%)", "w_sigma", 20.0, c2, min_value=0.01, step=1.0, format="%.2f")
    conv = sel(
        "Valor del dinero",
        "w_conv",
        ["Tasa continua", "Tasa discreta", "Dar B(0,T)"],
        help="Continua: B(t,T) = e^{-r(T-t)}.\n\n"
             "Discreta: B(t,T) = (1+r)^{-(T-t)}.\n\n"
             "También puedes capturar B(0,T) directamente."
    )
    if conv == "Dar B(0,T)":
        B0 = num("B(0,T)", "w_B0", 0.9512, min_value=0.0001, max_value=2.0, step=0.01, format="%.6f")
        r = -np.log(B0) / T * 100
    else:
        r = num("Tasa r (%)", "w_r", 5.0, min_value=-5.0, step=0.25, format="%.3f")

    st.subheader("Ingresos del subyacente")
    ingreso = sel("Tipo de ingreso", "w_ingreso", ["Sin ingresos", "Dividendos", "Ingreso fijo"], label_visibility="collapsed")
    q, pagos = 0.0, ()
    if ingreso == "Dividendos":
        q = num("Rendimiento q (%)", "w_q", 2.0, min_value=-5.0, step=0.25, format="%.3f") / 100
        st.caption("Da B_E(t,T):\n"
                   "A unidades hoy, serán B_E⁻¹(t,T)A unidades en T.")
    elif ingreso == "Ingreso fijo":
        tabla = tabla_editable("pagos", pd.DataFrame({"Fecha (años)": [0.25, 0.75], "Monto": [2.0, 2.0]}),
                               num_rows="dynamic", hide_index=True, width="stretch")
        limpio = tabla.dropna()
        pagos = tuple(sorted((float(a), float(b)) for a, b in zip(limpio["Fecha (años)"], limpio["Monto"]) if a > 0))

    st.divider()
    st.button("Restablecer datos", on_click=restablecer, width="stretch")

st.session_state["mercado"] = Mercado(S0=S0, r=float(r) / 100, sigma=sigma / 100, q=q, pagos=pagos,
                                      conv="discreta" if conv == "Tasa discreta" else "continua")
st.session_state["K"], st.session_state["T"] = K, T

paginas = {
    "": [st.Page(inicio.pagina, title="Inicio", url_path="inicio", default=True),
         st.Page(forwards.pagina, title="Forwards", url_path="forwards"),
         st.Page(arboles.pagina, title="Árboles", url_path="arboles"),
         st.Page(martingalas.pagina, title="Martingalas", url_path="martingalas"),
         st.Page(black_scholes.pagina, title="Black-Scholes", url_path="black-scholes"),
         st.Page(americanas.pagina, title="Americanas", url_path="americanas"),
         st.Page(arbitraje.pagina, title="Arbitraje", url_path="arbitraje")],
    "Calculadoras": [st.Page(opciones.pagina, title="Opciones", url_path="opciones"),
                     st.Page(exoticos.pagina, title="Exóticos", url_path="exoticos"),
                     st.Page(constructor.pagina, title="Constructor", url_path="constructor")],
    "Complementos": [st.Page(griegas.pagina, title="Griegas", url_path="griegas"),
                     st.Page(estrategias.pagina, title="Estrategias y riesgo", url_path="estrategias")],
}
actual = st.navigation(paginas, position="top")
if st.session_state.get("_pagina") != actual.url_path:
    st.session_state["_pagina"] = actual.url_path
    asentar_tablas()
actual.run()
