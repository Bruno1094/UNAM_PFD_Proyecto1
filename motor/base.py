"""Mercado, descuento e ingresos del subyacente.

Notación de las notas:
    B(t,T)   valor en t de $1 pagado en T  (tasa continua o discreta)
    B_E(t,T) proceso valor de los dividendos del subyacente
    C        valor presente de los ingresos fijos que paga el subyacente
    F        precio forward

Todo el motor trabaja internamente con tasas continuas equivalentes:
si la convención es discreta, r_c = ln(1 + r).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.stats import norm

N = norm.cdf
n = norm.pdf


@dataclass(frozen=True)
class Mercado:
    S0: float
    r: float
    sigma: float
    q: float = 0.0
    pagos: tuple = ()          # ((t_1, D_1), (t_2, D_2), ...)
    conv: str = "continua"     # "continua" | "discreta"

    # ---- tasas -----------------------------------------------------------
    @property
    def rc(self) -> float:
        return self.r if self.conv == "continua" else float(np.log1p(self.r))

    @property
    def qc(self) -> float:
        return self.q if self.conv == "continua" else float(np.log1p(self.q))

    @property
    def b(self) -> float:
        """Costo de acarreo r - q."""
        return self.rc - self.qc

    def B(self, t1, t2=None):
        """B(t1, t2); con un solo argumento, B(0, t1)."""
        plazo = t1 if t2 is None else t2 - t1
        return np.exp(-self.rc * plazo)

    def BE(self, t1, t2=None):
        plazo = t1 if t2 is None else t2 - t1
        return np.exp(-self.qc * plazo)

    # ---- ingresos --------------------------------------------------------
    def pagos_hasta(self, T, t=0.0):
        return [(ti, di) for ti, di in self.pagos if t < ti <= T + 1e-12]

    def vp_ingresos(self, T, t=0.0) -> float:
        """C: valor en t de los pagos en (t, T].

        Con q = 0 el descuento es B(t, t_i), como en las notas. Si además hay
        rendimiento continuo, se descuenta con e^{-(r-q)(t_i-t)} para que el
        forward siga libre de arbitraje.
        """
        return float(sum(di * np.exp(-self.b * (ti - t))
                         for ti, di in self.pagos_hasta(T, t)))

    def S_estrella(self, T):
        """Parte riesgosa del subyacente: S0 - C."""
        return self.S0 - self.vp_ingresos(T)

    def prepago(self, T):
        """Forward prepagado: S0·B_E(0,T) - C (con los ajustes anteriores)."""
        return self.S_estrella(T) * self.BE(T)

    def forward(self, T):
        """Precio forward F = prepago / B(0,T)."""
        return self.prepago(T) / self.B(T)

    def valor_forward(self, K, T):
        """f = prepago - K·B(0,T)."""
        return self.prepago(T) - K * self.B(T)

    # ---- utilidades ------------------------------------------------------
    def con(self, **kw) -> "Mercado":
        return replace(self, **kw)

    def avanzar(self, h: float) -> "Mercado":
        """El mismo mercado h años después (para theta)."""
        return self.con(pagos=tuple((t - h, d) for t, d in self.pagos if t - h > 0))
