import streamlit as st

from motor.vanilla import europea
from paginas.comun import K_T, caja, encabezado, mercado


def pagina():
    st.header("Realizado por:")
    izq, der = st.columns(2)
    izq.markdown("""
    
Hernández Alcántara Cristina Geraldine

Mendoza Aragón Rogelio

Moreno Ventura Miguel Angel
""")
    der.markdown("""
    
Olavarria Guerra Bruno

Reyes Díaz Axel Gabriel

Salgado Bahena Arturo
""")
