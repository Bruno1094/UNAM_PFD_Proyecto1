"""Árbol binomial del curso, con la notación de las sesiones 2 a 8.

Numeración de nodos: el nodo inicial es 1 y del nodo j salen 2j (abajo) y 2j+1 (arriba).
Una trayectoria se escribe ω = 0uud. El árbol se guarda siempre completo, así que un
payoff puede depender de la trayectoria (lookback, asiática, sin toque, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .simulacion import evaluar_expresion


# ---------------------------------------------------------------------------
# Construcción del árbol
# ---------------------------------------------------------------------------
@dataclass
class Arbol:
    N: int
    S: np.ndarray          # S[j], j = 1 .. 2^(N+1)-1
    B: np.ndarray          # B[j]: descuento del periodo que inicia en el nodo j

    @property
    def nodos(self):
        return 2 ** (self.N + 1) - 1


def tiempo(j: int) -> int:
    return int(np.floor(np.log2(j)))


def nivel(k: int):
    return np.arange(2 ** k, 2 ** (k + 1))


def nombre_nodo(j: int) -> str:
    """Nodo 11 -> 0duu."""
    return "0" + "".join("u" if c == "1" else "d" for c in bin(j)[3:])


def nodo_de(w: str) -> int:
    j = 1
    for c in w.replace("0", ""):
        j = 2 * j + (c == "u")
    return j


def _B_por_nodo(N, B):
    """B puede ser un número, una lista por fecha (N) o por nodo (2^N - 1)."""
    arr = np.atleast_1d(np.asarray(B, dtype=float))
    salida = np.zeros(2 ** N)
    if arr.size == 1:
        salida[1:] = arr[0]
    elif arr.size == N:
        for k in range(N):
            salida[nivel(k)] = arr[k]
    elif arr.size == 2 ** N - 1:
        salida[1:] = arr
    else:
        raise ValueError(f"Para B escribe un valor, {N} valores (uno por fecha) o {2 ** N - 1} (uno por nodo).")
    if (salida[1:] <= 0).any():
        raise ValueError("Los factores B deben ser positivos.")
    return salida


def multiplicativo(S0, u, d, N, B) -> Arbol:
    if u <= d:
        raise ValueError("u debe ser mayor que d.")
    S = np.zeros(2 ** (N + 1))
    S[1] = S0
    for j in range(1, 2 ** N):
        S[2 * j], S[2 * j + 1] = d * S[j], u * S[j]
    return Arbol(N, S, _B_por_nodo(N, B))


def por_fechas(tabla, B) -> Arbol:
    """Recombinante: tabla[k] tiene k+1 precios, de abajo hacia arriba."""
    N = len(tabla) - 1
    for k, fila in enumerate(tabla):
        if len(fila) != k + 1:
            raise ValueError(f"En la fecha {k} debe haber {k + 1} precios y hay {len(fila)}.")
    S = np.zeros(2 ** (N + 1))
    for k in range(N + 1):
        for j in nivel(k):
            subidas = bin(int(j))[3:].count("1")
            S[j] = tabla[k][subidas]
    return Arbol(N, S, _B_por_nodo(N, B))


def por_nodos(precios, B) -> Arbol:
    """General: S1, S2, S3, ... en el orden de la numeración de clase."""
    M = len(precios)
    N = int(np.log2(M + 1)) - 1
    if 2 ** (N + 1) - 1 != M or N < 1:
        raise ValueError("Hacen falta 3, 7, 15 o 31 precios: uno por cada nodo del árbol.")
    S = np.zeros(2 ** (N + 1))
    S[1:] = precios
    return Arbol(N, S, _B_por_nodo(N, B))


def parametros(metodo, r, sigma, delta, mu=None):
    """Elección de u y d de la sesión 7. En el curso se usa μ = r - σ²/2."""
    m = r - 0.5 * sigma ** 2 if metodo == "curso" else mu
    raiz = sigma * np.sqrt(delta)
    return float(np.exp(m * delta + raiz)), float(np.exp(m * delta - raiz)), float(np.exp(-r * delta))


# ---------------------------------------------------------------------------
# Payoff como función de la trayectoria
# ---------------------------------------------------------------------------
def caminos(a: Arbol, k: int):
    """Matriz (2^k, k+1) con el precio en cada fecha para los nodos de la fecha k."""
    finales = nivel(k)
    return np.column_stack([a.S[finales >> (k - i)] for i in range(k + 1)])


def pago(a: Arbol, k: int, expresion: str, **constantes):
    """Evalúa X en los nodos de la fecha k.

    Variables: S (actual), S0, A y G (promedios desde S(0)), M (máximo), m (mínimo),
    Sm (precio a la mitad del horizonte) y las constantes que se pasen.
    """
    t = caminos(a, k)
    with np.errstate(all="ignore"):
        env = {"S": t[:, -1], "S0": a.S[1], "A": t.mean(axis=1), "G": np.exp(np.log(t).mean(axis=1)),
               "M": t.max(axis=1), "m": t.min(axis=1), "Sm": t[:, min(a.N // 2, k)], **constantes}
        return np.broadcast_to(np.asarray(evaluar_expresion(expresion, env), dtype=float).astype(float), (2 ** k,)).copy()


# ---------------------------------------------------------------------------
# Valuación hacia atrás
# ---------------------------------------------------------------------------
@dataclass
class Valuacion:
    a: Arbol
    q: np.ndarray
    V: np.ndarray          # valor h_k en cada nodo
    X: np.ndarray          # valor intrínseco (0 si es europea, antes de T)
    cont: np.ndarray       # valor de dejarla vivir
    alfa: np.ndarray
    beta: np.ndarray
    ejercer: np.ndarray
    estilo: str = "europea"
    problemas: list = field(default_factory=list)

    @property
    def precio(self):
        return float(self.V[1])


def valuar(a: Arbol, expresion: str, estilo="europea", **constantes) -> Valuacion:
    """V_n = B[q V_u + (1-q) V_d], y en la americana h_k = máx{continuación, X}."""
    n = 2 ** (a.N + 1)
    q, V, X, cont, alfa, beta = (np.full(n, np.nan) for _ in range(6))
    ejercer = np.zeros(n, bool)
    finales = nivel(a.N)
    V[finales] = X[finales] = pago(a, a.N, expresion, **constantes)
    problemas = []
    for k in range(a.N - 1, -1, -1):
        js = nivel(k)
        Sn, Sd, Su = a.S[js], a.S[2 * js], a.S[2 * js + 1]
        if (Su == Sd).any():
            raise ValueError("Dos sucesores de un nodo tienen el mismo precio: el activo no sería riesgoso.")
        B = a.B[js]
        q[js] = (Sn / B - Sd) / (Su - Sd)
        Vu, Vd = V[2 * js + 1], V[2 * js]
        cont[js] = B * (q[js] * Vu + (1 - q[js]) * Vd)
        alfa[js] = (Vu - Vd) / (Su - Sd)
        beta[js] = B * (Vu - alfa[js] * Su)
        if estilo == "americana":
            X[js] = pago(a, k, expresion, **constantes)
            ejercer[js] = X[js] > cont[js] + 1e-12
            V[js] = np.maximum(cont[js], X[js])
        else:
            V[js] = cont[js]
        malos = js[(q[js] <= 0) | (q[js] >= 1)]
        problemas += [int(j) for j in malos]
    return Valuacion(a, q, V, X, cont, alfa, beta, ejercer, estilo, sorted(problemas))


def seguir(v: Valuacion, w: str) -> list:
    """Sigue la cobertura sobre ω y muestra que se replica y se autofinancia."""
    a = v.a
    j = 1
    filas = []
    previo = None
    for k in range(a.N + 1):
        fila = {"Fecha": f"{k}δ", "Nodo": j, "ω": nombre_nodo(j), "S": a.S[j], "Valor del derivado": v.V[j]}
        if previo is not None:
            al, be, Bp = previo
            fila["Cartera anterior vale"] = al * a.S[j] + be / Bp
        if k < a.N:
            fila["α nueva"], fila["β nueva"] = v.alfa[j], v.beta[j]
            fila["Cartera nueva cuesta"] = v.alfa[j] * a.S[j] + v.beta[j]
            if v.estilo == "americana":
                fila["Consumo si no ejerce"] = v.V[j] - v.cont[j]
            previo = (v.alfa[j], v.beta[j], a.B[j])
            j = 2 * j + (w[k] == "u")
        filas.append(fila)
    return filas


def formula_binomial(S0, u, d, N, B, expresion, **constantes):
    """X(0) = B^N Σ C(N,j) q^j (1-q)^(N-j) X(S(0) u^j d^(N-j)). Solo para X(S(T))."""
    from math import comb
    q = (1 / B - d) / (u - d)
    filas = []
    total = 0.0
    for j in range(N + 1):
        ST = S0 * u ** j * d ** (N - j)
        X = float(np.ravel(evaluar_expresion(expresion, {"S": np.array([ST]), "S0": S0, **constantes}))[0])
        prob = comb(N, j) * q ** j * (1 - q) ** (N - j)
        total += prob * X
        filas.append({"Subidas j": j, "C(N,j)": comb(N, j), "S_T": ST, "Probabilidad Q": prob, "X": X})
    return B ** N * total, q, filas


# ---------------------------------------------------------------------------
# Filtración, esperanza condicional y martingalas
# ---------------------------------------------------------------------------
def probabilidad_p(a: Arbol, p) -> np.ndarray:
    """p por nodo; acepta un número o una lista por nodo no final."""
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    salida = np.zeros(2 ** a.N)
    if arr.size == 1:
        salida[1:] = arr[0]
    elif arr.size == 2 ** a.N - 1:
        salida[1:] = arr
    else:
        raise ValueError(f"Escribe una probabilidad, o {2 ** a.N - 1} (una por nodo no final).")
    if ((salida[1:] < 0) | (salida[1:] > 1)).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")
    return salida


def filtracion(N: int, n: int) -> list:
    """F(n) como las trayectorias parciales de nodos: {1,2,4}."""
    return [[int(j >> (n - i)) for i in range(n + 1)] for j in nivel(n)]


def esperanza_condicional(a: Arbol, finales, prob) -> np.ndarray:
    """E[X | F(n)] en cada nodo, trabajando hacia atrás."""
    E = np.full(2 ** (a.N + 1), np.nan)
    E[nivel(a.N)] = finales
    for k in range(a.N - 1, -1, -1):
        js = nivel(k)
        E[js] = prob[js] * E[2 * js + 1] + (1 - prob[js]) * E[2 * js]
    return E


def es_martingala(a: Arbol, proceso, prob, tol=1e-9):
    """Compara cada nodo con el valor esperado del siguiente paso."""
    filas = []
    for k in range(a.N):
        for j in nivel(k):
            esp = prob[j] * proceso[2 * j + 1] + (1 - prob[j]) * proceso[2 * j]
            filas.append({"Nodo": int(j), "ω": nombre_nodo(int(j)), "Y": proceso[j], "E[siguiente | F]": esp,
                          "Cumple": bool(abs(esp - proceso[j]) < tol)})
    return all(f["Cumple"] for f in filas), filas


def descuento_acumulado(a: Arbol) -> np.ndarray:
    """B(0, nδ) a lo largo de cada nodo."""
    D = np.ones(2 ** (a.N + 1))
    for j in range(2, 2 ** (a.N + 1)):
        D[j] = D[j // 2] * a.B[j // 2]
    return D


def representacion(v: Valuacion):
    """Z = B·S, Y = E_Q[B(0,Nδ) X | F(n)] y α = δY/δZ del teorema de representación."""
    a = v.a
    D = descuento_acumulado(a)
    Z = D * a.S
    Z[0] = np.nan
    finales = nivel(a.N)
    Y = esperanza_condicional(a, D[finales] * v.V[finales], v.q)
    alfa = np.full(2 ** a.N, np.nan)
    for j in range(1, 2 ** a.N):
        alfa[j] = (Y[2 * j + 1] - Y[2 * j]) / (Z[2 * j + 1] - Z[2 * j])
    return Z, Y, alfa


def trayectoria_prob(a: Arbol, prob) -> np.ndarray:
    """Probabilidad de llegar a cada nodo (τ en los nodos finales)."""
    t = np.ones(2 ** (a.N + 1))
    for j in range(1, 2 ** a.N):
        t[2 * j + 1] = t[j] * prob[j]
        t[2 * j] = t[j] * (1 - prob[j])
    return t


def radon_nikodym(v: Valuacion, p) -> dict:
    a = v.a
    tau, tau_q = trayectoria_prob(a, p), trayectoria_prob(a, v.q)
    finales = nivel(a.N)
    tau_q[0] = tau[0] = np.nan
    pos_p, pos_q = tau[finales] > 0, tau_q[finales] > 0
    equivalentes = bool((pos_p == pos_q).all() and (tau_q[finales] >= 0).all())
    with np.errstate(all="ignore"):
        zeta = tau_q / tau
    return {"tau": tau, "tau_q": tau_q, "zeta": zeta, "equivalentes": equivalentes}


# ---------------------------------------------------------------------------
# Tiempos de paro (sesión 8)
# ---------------------------------------------------------------------------
def todas_trayectorias(N):
    return ["".join("u" if c == "1" else "d" for c in format(i, f"0{N}b")) for i in range(2 ** N)]


def tiempo_optimo(v: Valuacion) -> dict:
    """τ = mín{k : h_k(S(kδ)) = X(S(kδ))}."""
    salida = {}
    for w in todas_trayectorias(v.a.N):
        j = 1
        for k in range(v.a.N + 1):
            if k == v.a.N or abs(v.V[j] - v.X[j]) < 1e-12:
                salida[w] = k
                break
            j = 2 * j + (w[k] == "u")
    return salida


def tiempo_minimo(a: Arbol) -> dict:
    """ρ = primer tiempo en que S alcanza el mínimo de toda la trayectoria."""
    salida = {}
    for w in todas_trayectorias(a.N):
        j, valores = 1, [a.S[1]]
        for c in w:
            j = 2 * j + (c == "u")
            valores.append(a.S[j])
        salida[w] = int(np.argmin(valores))
    return salida


def es_tiempo_de_paro(tau: dict) -> tuple:
    """{τ = k} debe poder decidirse con los primeros k movimientos."""
    violaciones = []
    for k in sorted(set(tau.values())):
        conjunto = {w for w, t in tau.items() if t == k}
        for w in conjunto:
            for otro in tau:
                if otro[:k] == w[:k] and otro not in conjunto:
                    violaciones.append((k, w, otro))
    return not violaciones, violaciones


def atomos(tau: dict) -> list:
    """Átomos de F(τ): trayectorias que coinciden hasta τ."""
    grupos = {}
    for w, t in tau.items():
        grupos.setdefault((t, w[:t]), []).append(w)
    return [sorted(g) for g in grupos.values()]


def muestreo_opcional(v: Valuacion, tau: dict) -> list:
    """Z martingala: Z(τ) = E_Q[Z(N) | F(τ)] en cada átomo."""
    a = v.a
    D = descuento_acumulado(a)
    Z = D * a.S
    E = esperanza_condicional(a, Z[nivel(a.N)], v.q)
    filas = []
    for grupo in atomos(tau):
        w = grupo[0]
        t = tau[w]
        j = nodo_de(w[:t])
        filas.append({"Átomo": ", ".join("0" + g for g in grupo), "τ": t, "Z(τ)": Z[j], "E_Q[Z(Nδ) | F(τ)]": E[j]})
    return filas


# ---------------------------------------------------------------------------
# Carrera de caballos (sesión 1)
# ---------------------------------------------------------------------------
def carrera(p_A, pago_carrera, pago_boleto, precio_boleto) -> dict:
    """La carrera paga `pago_carrera` por peso si gana A. El boleto paga `pago_boleto` si gana A."""
    esperado = p_A * pago_boleto
    sostenible = pago_boleto / pago_carrera
    dif = precio_boleto - sostenible
    if abs(dif) < 1e-12:
        piernas = []
    elif dif > 0:
        piernas = [{"Operación": "Vender el boleto", "Hoy": precio_boleto, "Si gana A": -pago_boleto, "Si gana B": 0.0},
                   {"Operación": f"Apostar {sostenible:,.2f} a la carrera", "Hoy": -sostenible, "Si gana A": pago_boleto, "Si gana B": 0.0}]
    else:
        piernas = [{"Operación": "Comprar el boleto", "Hoy": -precio_boleto, "Si gana A": pago_boleto, "Si gana B": 0.0},
                   {"Operación": f"Tomar una apuesta por {sostenible:,.2f}", "Hoy": sostenible, "Si gana A": -pago_boleto, "Si gana B": 0.0}]
    return {"esperado": esperado, "sostenible": sostenible, "ganancia": abs(dif), "piernas": piernas,
            "ganancia_media": esperado - precio_boleto}


# ---------------------------------------------------------------------------
# Fórmula (5) de la sesión 7
# ---------------------------------------------------------------------------
def valuar_lognormal(S0, r, sigma, T, expresion, **constantes) -> float:
    """X(0) = e^{-rT} E[X(S(0)e^Y)], Y ~ N((r - σ²/2)T, σ²T), por cuadratura."""
    z = np.linspace(-12, 12, 2_000_001)
    dens = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    X = np.broadcast_to(np.asarray(evaluar_expresion(expresion, {"S": S, "S0": S0, **constantes}), dtype=float), S.shape)
    return float(np.exp(-r * T) * np.trapezoid(X * dens, z))


# ---------------------------------------------------------------------------
# Ejemplos de clase
# ---------------------------------------------------------------------------
EJEMPLOS = {
    "Sesión 3: ejemplo con call K = 100": dict(modo="fechas", tabla="100\n80, 120\n60, 100, 140\n40, 80, 120, 160", B="1", pago="Call", K=100.0, estilo="Europea", w="uud"),
    "Sesión 3, ejercicio 2: digital": dict(modo="fechas", tabla="100\n80, 120\n60, 100, 140\n40, 80, 120, 160", B="1", pago="Fórmula propia", expr="100*(S>S0)", K=100.0, estilo="Europea", w="uud"),
    "Sesión 3, ejercicio 3: call K = 60": dict(modo="fechas", tabla="60\n50, 70\n40, 60, 80", B="1", pago="Call", K=60.0, estilo="Europea", w="ud"),
    "Sesión 4, ejercicio 2: call K = 115": dict(modo="mult", param="Dar u y d", S0=120.0, u=1.7, d=0.8, N=3, B=f"{np.exp(-0.06):.10f}", pago="Call", K=115.0, estilo="Europea", w="uud"),
    "Sesión 5: árbol de la martingala": dict(modo="nodos", precios="80, 60, 120, 36, 72, 80, 180", B="1", pago="Fórmula propia", expr="S", K=0.0, estilo="Europea", w="ud"),
    "Sesión 7, ejercicio 13: put K = 45": dict(modo="mult", param="Con σ y r (μ = r - ½σ²)", S0=50.0, sigma=.20, r=.056, T=1.0, N=2, pago="Put", K=45.0, estilo="Europea", w="ud"),
    "Sesión 7, ejercicio 14: call K = 100": dict(modo="mult", param="Con σ y r (μ = r - ½σ²)", S0=100.0, sigma=.30, r=.06, T=1.0, N=2, pago="Call", K=100.0, estilo="Europea", w="ud"),
    "Sesión 7, ejercicio 15: put con strike que se ajusta (2 periodos)": dict(modo="mult", param="Con σ y r (μ = r - ½σ²)", S0=40.0, sigma=.35, r=.05, T=1.0, N=2, pago="Fórmula propia", expr="max(si(Sm<35,35,40)-S,0)", K=40.0, estilo="Europea", w="ud"),
    "Sesión 7, ejercicio 15: put con strike que se ajusta (4 periodos)": dict(modo="mult", param="Con σ y r (μ = r - ½σ²)", S0=40.0, sigma=.35, r=.05, T=1.0, N=4, pago="Fórmula propia", expr="max(si(Sm<35,35,40)-S,0)", K=40.0, estilo="Europea", w="udud"),
    "Sesión 8: put americano K = 5": dict(modo="mult", param="Dar u y d", S0=4.0, u=2.0, d=0.5, N=2, B="0.8", pago="Put", K=5.0, estilo="Americana", w="du"),
    "Sesión 8, ejercicio 1: put K = 3": dict(modo="mult", param="Dar u y d", S0=4.0, u=3.0, d=0.5, N=2, B="0.5", pago="Put", K=3.0, estilo="Americana", w="ud"),
    "Sesión 8, ejercicio 2: put americano K = 60": dict(modo="mult", param="Con σ y r (μ = r - ½σ²)", S0=60.0, sigma=.35, r=.06, T=1.0, N=2, pago="Put", K=60.0, estilo="Americana", w="ud"),
    "Sesión 8, ejercicio 3: call asiática": dict(modo="mult", param="Dar u y d", S0=4.0, u=2.0, d=0.5, N=3, B="0.8", pago="Asiática call", K=4.0, estilo="Europea", w="udu"),
    "Clase: lookback del máximo, 5 meses": dict(modo="mult", param="Dar u y d", S0=100.0, u=1.2, d=0.9, N=5, B=f"{np.exp(-0.05 / 12):.10f}", pago="Lookback: máximo", K=0.0, estilo="Europea", w="ududu"),
    "Clase: doble sin toque, 2 años": dict(modo="mult", param="Dar u y d", S0=100.0, u=1.4, d=0.7, N=4, B=f"{np.exp(-0.02):.10f}", pago="Doble sin toque", K=0.0, B1=50.0, B2=250.0, monto=100.0, estilo="Europea", w="udud"),
}
