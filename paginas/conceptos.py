import pandas as pd
import streamlit as st

from motor import curso as cu
from paginas.comun import aviso, caja, encabezado, num, tabla

TEMAS = [
    ("Sesión 1", "Incertidumbre, arbitraje y cobertura", r"""
**Incertidumbre.** No se conoce el valor particular que tomará un activo, pero sí los valores que puede tomar.

**Subyacente.** Activo cuyo precio presenta incertidumbre.

**Derivado.** Instrumento cuyo valor depende del valor de una o más variables o parámetros, sus subyacentes.

**Arbitraje.** Asegurar una ganancia sin riesgo con transacciones simultáneas en dos o más mercados.

**Cobertura.** Estrategia que se comporta como el contingente sin ser el contingente. Replica su pago en cualquier escenario.

**Precio de un contingente.** Valor con el que no se puede construir una estrategia de arbitraje.

**Precio correcto o sostenible.** El precio de su cobertura.

**Principio de no arbitraje.** Si dos cosas tienen el mismo valor en una fecha futura, sus valores de hoy deben ser iguales.

**Supuestos.** Competencia perfecta, información simétrica y completa, y agentes que pueden tomar cualquiera de las dos posiciones.
"""),
    ("Sesión 1", "Forwards y opciones", r"""
**Forward.** Acuerdo para comprar o vender el subyacente a un precio pactado en una fecha futura. La parte larga se obliga a comprar y la corta a vender.

**Payoff.** Valor de una posición cuando se cumplen las condiciones del contrato. Forward largo: $S_T-K$; corto: $K-S_T$. Es un juego de suma cero.

**$B(t,T)$.** Valor en $t$ de un peso pagado en $T$.

**Valor del forward.** $f_t=S_t-B(t,T)K$.

**Precio forward.** El $K$ que hace que el contrato valga cero: $F_t=B^{-1}(t,T)S_t$. Es independiente del modelo: no depende de la dinámica del subyacente.

**Con ingreso fijo.** $C$ es el valor presente del ingreso: $f_t=S_t-C-KB(t,T)$ y $F_t=(S_t-C)B^{-1}(t,T)$.

**Con dividendos.** $f_0=S_0B_E(0,T)-KB(0,T)$ y $F_0=S_0B_E(0,T)B^{-1}(0,T)$.

**Opción.** Da el derecho, mas no la obligación, de comprar (call) o vender (put). La europea se ejerce en $T$; la americana, en cualquier momento.

**Paridad put-call.** $c(S_0,T,K)-p(S_0,T,K)=S_0-B(0,T)K$. Un call largo y un put corto igualan un forward largo.
"""),
    ("Sesión 2", "Modelo binomial de un periodo", r"""
**Idea.** Construir de manera sintética el derivado con $\alpha$ unidades de subyacente y $\beta$ en la cuenta en efectivo.

**Cuenta en efectivo.** Activo libre de riesgo. No presenta incertidumbre.

**Mercado completo.** Todo derivado es replicable, así que todo derivado tiene un precio de no arbitraje.

**Medida de riesgo neutral.** $Q=\{q,1-q\}$, con $q=\dfrac{S_0B^{-1}(0,T)-S_d}{S_u-S_d}$.

**Sin arbitraje.** $0<q<1\iff S_d<S_0B^{-1}(0,T)<S_u$.

**Precio.** $V_0=B(0,T)E_Q[X]$. No depende de las probabilidades reales ni de la actitud ante el riesgo.
"""),
    ("Sesiones 3 y 4", "Modelo multiperiodo", r"""
**Numeración.** $0,1\delta,\dots,N\delta=T$. Del nodo $j$ salen $2j$ y $2j+1$, con $2^N$ nodos finales.

**Recombina.** $S_{ud}=S_{du}$: quedan $N+1$ nodos finales.

**Multiplicativo.** Existen $u,d$ con $S_u=uS_n$ y $S_d=dS_n$. Todas las $q$ de un mismo tiempo son iguales. Todo multiplicativo recombina, pero no al revés.

**Trayectoria.** Se fija $\omega=0\omega_1\cdots\omega_N$ con $\omega_j\in\{u,d\}$.

**Estrategia dinámica.** La cartera se ajusta en cada periodo con la $\alpha$ y $\beta$ del nodo.

**Autofinanciada.** No hacen falta recursos adicionales ni sobran.

**Derivado trivial.** $X=S(T)$ vale $S(0)$, de donde $B(0,N\delta)E_Q[S(T)]=S(0)$.

**Libre de modelo.** Instrumento que se valúa sin suponer una dinámica para el subyacente, como el forward.
"""),
    ("Sesión 5", "Filtraciones y martingalas", r"""
**Espacio de probabilidad.** $(\Omega,F,P)$, donde $P$ es la medida de ocurrencia real.

**Filtración.** Familia creciente de σ-álgebras $F(n)\subset F(n+1)$. Describe cómo se acumula la información.

**Proceso adaptado.** $S(n\delta)$ es $F(n)$-medible.

**Payoff.** Variable aleatoria definida en los nodos finales.

**Proceso previsible.** Sus valores en $n\delta$ quedan determinados por la información hasta $(n-1)\delta$.

**Martingala.** $E_P[Y(i\delta)\mid F(j)]=Y(j\delta)$ para $i\ge j$: el proceso no tiene tendencia.

**Propiedad de torre.** $E[E[Y\mid F(i)]\mid F(j)]=E[Y\mid F(j)]$ para $j\le i$. Por eso $E_P[X\mid F(n)]$ siempre es martingala.

**Teorema de representación binomial.** Si $Z$ y $Y$ son Q-martingalas, existe $\alpha$ previsible tal que $Y(n\delta)=Y(0)+\sum_{k=1}^n\alpha(k\delta)\Delta Z(k\delta)$.
"""),
    ("Sesión 6", "Justificación del precio y Radon-Nikodym", r"""
**Identificación.** $Z(n\delta)=B(0,n\delta)S(n\delta)$ representa al subyacente, $Y(n\delta)=E_Q[B(0,N\delta)X\mid F(n)]$ al derivado y $\alpha$ a la cantidad de subyacente.

**Portafolio.** $\pi(n)$ guarda $\alpha((n+1)\delta)$ títulos y $\beta((n+1)\delta)=Y(n\delta)-\alpha((n+1)\delta)Z(n\delta)$.

**Resultado.** La estrategia se autofinancia y replica $X$. Por eso el precio es $E_Q[B(0,N\delta)X]$.

**Valor en $n\delta$.** $B^{-1}(0,n\delta)E_Q[B(0,N\delta)X\mid F(n)]$.

**Derivada de Radon-Nikodym.** $\dfrac{dQ}{dP}(i)=\dfrac{\hat\tau(i)}{\tau(i)}$, el cociente de probabilidades de trayectoria.

**Medidas equivalentes.** $P(A)>0\iff Q(A)>0$. Es necesario para que exista $dQ/dP$.

**Proceso de Radon-Nikodym.** $\zeta(j\delta)=E_P[dQ/dP\mid F(j)]$, la cantidad de cambio de medida hasta $j\delta$.
"""),
    ("Sesión 7", "Límite y Black-Scholes", r"""
**Tasa equivalente.** $r(j\delta)\delta=\ln S((j+1)\delta)-\ln S(j\delta)$.

**Árbol.** Retornos con media $\mu\delta$ y varianza $\sigma^2\delta$: $u=e^{\mu\delta+\sigma\sqrt\delta}$ y $d=e^{\mu\delta-\sigma\sqrt\delta}$.

**Lognormal.** Por el Teorema del Límite Central, $\ln S(t_2)-\ln S(t_1)\sim N(\mu(t_2-t_1),\sigma^2(t_2-t_1))$, con incrementos independientes.

**$\mu$ y $\sigma$.** Retorno esperado y volatilidad. El precio no depende de $\mu$.

**Elección del curso.** $\mu=r-\frac12\sigma^2$.
"""),
    ("Sesión 8", "Opciones americanas", r"""
**Notación.** $X(n\delta)$ es el valor intrínseco y $V(n\delta)$ el valor del derivado. Debe cumplirse $V(k\delta)\ge X(S(k\delta))$.

**Ejercicio óptimo.** Ejercer al llegar a un nodo donde $h_k$ coincide con el valor intrínseco.

**Consumo.** Si conviene ejercer y el poseedor no lo hace, el emisor puede consumir la diferencia para mantener $V(k\delta)=h_k(S(k\delta))$.

**Tiempo de paro.** $\tau$ con $\{\tau=k\}\in F(k)$ para toda $k$.

**$F(\tau)$.** σ-álgebra de los conjuntos determinados por $\tau$: $A\cap\{\tau=k\}\in F(k)$.

**Muestreo opcional.** Si $Z$ es martingala y $\tau\le\rho$ están acotados, $Z(\tau)=E[Z(\rho)\mid F(\tau)]$.
"""),
    ("Apuntes", "Otros términos de clase", r"""
**Apostadores y tomadores.** Los primeros entregan dinero al inicio; los segundos lo reciben.

**Price taker.** Participante que no puede mover el precio.

**Step-in y unwind.** Tomar una posición y salir de ella.

**Exóticos vistos.** Digital, lookback ($X=\max\{S(0),\dots,S(N\delta)\}$), asiática y doble sin toque ($X=100\cdot\mathbb 1\{B_1<S(t)<B_2\}$).

**Bibliografía.** Baxter y Rennie.
"""),
]


def pagina():
    encabezado("Conceptos", "Las ideas de clase, en el orden de las sesiones.")
    t1, t2 = st.tabs(["Definiciones", "Carrera de caballos"])
    with t1:
        buscar = st.text_input("Buscar un concepto", key="c_buscar", placeholder="Por ejemplo: martingala").strip().lower()
        vistos = 0
        for ses, titulo, texto in TEMAS:
            bloques = [b for b in texto.strip().split("\n\n") if not buscar or buscar in b.lower()]
            if bloques:
                vistos += 1
                with st.expander(f"{ses}: {titulo}", expanded=bool(buscar)):
                    st.markdown("\n\n".join(bloques))
        if not vistos:
            st.caption("Sin resultados. Prueba con otra palabra.")

    with t2:
        st.caption("Sesión 1. Muestra por qué la esperanza no da el precio: la cobertura sí.")
        c = st.columns(4)
        p = num("Probabilidad de que gane A", "w_c_pA", .70, c[0], min_value=0.0, max_value=1.0, step=.05)
        pc = num("La carrera paga, por peso", "w_c_pc", 2.0, c[1], min_value=.01, step=.5)
        pb = num("El boleto paga si gana A", "w_c_pb", 100.0, c[2], min_value=0.0, step=10.0)
        pr = num("Precio del boleto", "w_c_pr", 60.0, c[3], min_value=0.0, step=5.0)
        r = cu.carrera(p, pc, pb, pr)
        c = st.columns(3)
        caja("Precio sostenible", r["sostenible"], "el costo de la cobertura", cont=c[0])
        caja("Pago esperado del boleto", r["esperado"], "ley de los grandes números", True, cont=c[1])
        caja("Ganancia media por boleto", r["ganancia_media"], "al comprarlo a ese precio, solo en promedio", True, cont=c[2])
        if r["piernas"]:
            aviso("Hay arbitraje", f"Con esta estrategia te quedas con {r['ganancia']:,.2f} sin importar quién gane.")
            tabla(pd.DataFrame(r["piernas"]), 2)
        else:
            aviso("El precio es sostenible", "No existe una estrategia para ganar siempre.", "neutro")
        st.caption("El precio sostenible no usa la probabilidad de que gane A. Cambia ese dato y compruébalo.")
