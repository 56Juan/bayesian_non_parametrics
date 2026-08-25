"""
progreso
========
Salida de progreso por consola para los bucles largos del proyecto.

Por que existe este modulo
--------------------------
Varias rutinas de `fit/` y `graphics/` recorren el producto cartesiano
(componente FPCA x parametro x variable x cadena) o (ventana x metrica), con
diagnosticos costosos dentro --`ess_geyer` es un bucle Python sobre lags, el
CRPS muestral recorre S extracciones por ventana--. Sin ninguna senal, quien
ejecuta el notebook no distingue "esta trabajando" de "se colgo", y la unica
salida del proyecto son ficheros y figuras que aparecen al final.

La convencion la fija `psbp_fd_v1/v2`, que ya imprimian progreso con un
`verbose_every` y un `print` a secas. Se replica esa idea aqui, en un solo
lugar, en vez de repartir `print` sueltos por cada modulo: el formato de la
linea de progreso es una convencion del proyecto y duplicarla garantiza que se
desalinee. Mismo criterio que con `pesos_trapezoidales` y `safe_chol`.

Decisiones
----------
- **Sin dependencias nuevas.** Nada de `tqdm`: obligaria a tocar `pyproject`, y
  en bucles anidados como `tabla_diagnosticos` las barras se pisan entre si.
  Un `print` por hito es legible tanto en notebook como en consola.
- **`verbose=False` por defecto en toda la API.** El silencio es el contrato
  vigente de estas funciones; activarlo por omision cambiaria la salida de los
  notebooks ya ejecutados. Quien quiera ver el progreso lo pide.
- **`flush=True` siempre.** Sin el, el buffer de stdout retiene las lineas y
  aparecen todas juntas al terminar, que es exactamente lo que se queria
  evitar.
- **El progreso no altera el resultado.** Ninguna funcion de este modulo
  devuelve algo que entre en un calculo; si se desactiva, los numeros son
  identicos bit a bit.

Uso
---
    p = Progreso("tabla_diagnosticos", total=n_items, verbose=verbose)
    for ...:
        p.paso(f"fpc_{fpc} beta_j {etiqueta}")
    p.fin()

`paso()` imprime como mucho `cada` veces por bucle (mas el primero y el
ultimo), de modo que el volumen de salida no depende del tamano del problema.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

__all__ = ["Progreso", "aviso"]


def aviso(mensaje: str, verbose: bool = True) -> None:
    """Imprime una linea suelta de contexto. No hace nada si `verbose` es False."""
    if verbose:
        print(mensaje, flush=True)


class Progreso:
    """
    Contador de progreso con submuestreo de la salida y tiempo transcurrido.

    nombre  : etiqueta de la rutina, aparece en cada linea.
    total   : numero de pasos esperados. Si es None se omite el porcentaje y el
              tiempo restante estimado, porque ambos requieren conocer el final.
    verbose : si es False el objeto queda inerte y `paso()` solo incrementa un
              contador. Se construye igual para no llenar el codigo llamante de
              condicionales.
    cada    : numero maximo de lineas intermedias a emitir. Con el valor por
              defecto un bucle de 10 items imprime los 10 y uno de 100000
              imprime 20, de modo que la salida es acotada sin necesidad de
              ajustar el parametro por caso.
    """

    def __init__(self, nombre: str, total: Optional[int] = None,
                 verbose: bool = False, cada: int = 20):
        self.nombre = str(nombre)
        self.total = int(total) if total is not None else None
        self.verbose = bool(verbose)
        self.n = 0
        self._t0 = time.perf_counter()

        # Cada cuantos pasos se emite una linea. Con `total` conocido se reparte
        # el presupuesto de `cada` lineas a lo largo del bucle; sin el, se cae a
        # un intervalo fijo porque no hay nada que repartir.
        if self.total is not None and cada > 0:
            self._intervalo = max(1, self.total // max(1, cada))
        else:
            self._intervalo = 1 if cada <= 0 else int(cada)

        if self.verbose:
            cola = f" ({self.total} pasos)" if self.total is not None else ""
            print(f"[{self.nombre}] inicio{cola}", flush=True)

    # ------------------------------------------------------------------
    def paso(self, detalle: str = "") -> None:
        """Registra un paso y, si toca segun el submuestreo, lo imprime."""
        self.n += 1
        if not self.verbose:
            return
        ultimo = self.total is not None and self.n >= self.total
        if self.n == 1 or ultimo or self.n % self._intervalo == 0:
            print(f"  {self._linea(detalle)}", flush=True)

    # ------------------------------------------------------------------
    def fin(self, detalle: str = "") -> None:
        """Cierra el bucle informando total de pasos y tiempo de pared."""
        if not self.verbose:
            return
        t = time.perf_counter() - self._t0
        cola = f"  {detalle}" if detalle else ""
        print(f"[{self.nombre}] listo: {self.n} pasos en {self._fmt(t)}{cola}",
              flush=True)

    # ------------------------------------------------------------------
    def _linea(self, detalle: str) -> str:
        t = time.perf_counter() - self._t0
        if self.total:
            pct = 100.0 * self.n / self.total
            # ETA por extrapolacion lineal del ritmo observado. Es grosera
            # cuando el costo por paso varia, pero basta para decidir si
            # conviene esperar o interrumpir, que es para lo que se mira.
            resto = (t / self.n) * (self.total - self.n) if self.n else 0.0
            cab = (f"{self.n}/{self.total} ({pct:4.1f}%) "
                   f"t={self._fmt(t)} eta={self._fmt(resto)}")
        else:
            cab = f"{self.n} t={self._fmt(t)}"
        return f"{cab}  {detalle}".rstrip()

    @staticmethod
    def _fmt(segundos: float) -> str:
        if segundos < 60:
            return f"{segundos:.1f}s"
        if segundos < 3600:
            return f"{segundos / 60:.1f}min"
        return f"{segundos / 3600:.2f}h"
