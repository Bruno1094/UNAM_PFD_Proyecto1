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
