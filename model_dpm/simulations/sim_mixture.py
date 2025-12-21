import numpy as np
from scipy.stats import laplace, norm

def multimodal_mixture_sampler(components):
    """
    Genera datos de una mezcla de distribuciones (Laplace o Normal)
    según los parámetros definidos en 'components'.
    """
    x = []
    for comp in components:
        # Validación de parámetros
        if 'loc' not in comp or 'scale' not in comp or 'size' not in comp:
            raise ValueError(f"Faltan parámetros en 'comp': {comp}")
        
        # Generación de muestras según el tipo de distribución
        if comp['type'] == 'laplace':
            x.append(laplace.rvs(loc=comp['loc'], scale=comp['scale'], size=comp['size']))
        elif comp['type'] == 'normal':
            x.append(norm.rvs(loc=comp['loc'], scale=comp['scale'], size=comp['size']))
        else:
            raise ValueError(f"Tipo de distribución desconocido: {comp['type']}")

    # Concatenación eficiente de las muestras
    return np.concatenate(x)

