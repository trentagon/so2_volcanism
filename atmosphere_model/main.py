from matplotlib import pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

import os
THISFILE = os.path.dirname(os.path.abspath(__file__))

try:
    from .models import AdiabatClimateEquilibrium
except ImportError:
    from models import AdiabatClimateEquilibrium

_CLIMATE_MODEL = None

def _build_default_climate_model():
    return AdiabatClimateEquilibrium(
        species_file=os.path.join(THISFILE, "input/species_climate.yaml"),
        settings_file=os.path.join(THISFILE, "input/settings_climate.yaml"),
        flux_file=os.path.join(THISFILE, "input/gj176_scaled_to_l9859b.txt"),
        thermo_file=os.path.join(THISFILE, "input/thermo.yaml"),
    )

def get_climate_model():
    """Return the shared default climate model instance."""
    global _CLIMATE_MODEL
    if _CLIMATE_MODEL is None:
        _CLIMATE_MODEL = _build_default_climate_model()
    return _CLIMATE_MODEL

def plot(P, T, mix, ylim, filename, P_ref=1e3, input_mix=None):
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1,1,figsize=[5,4])

    ind = np.argmin(np.abs(P - P_ref))
    species_layer = []
    mix_layer = np.empty(len(mix))
    for i,sp in enumerate(mix):
        species_layer.append(sp)
        mix_layer[i] = mix[sp][ind]
    inds = np.argsort(mix_layer)[::-1]
    
    species_colors = {}
    for i in inds[:10]:
        sp = species_layer[i]
        line, = ax.plot(mix[sp], P/1e6, lw=2, label=sp)
        species_colors[sp] = line.get_color()

    # Optional vertical guides for input SO2/CO2/H2O mixing ratios.
    if input_mix is not None:
        guide_species = ['SO2', 'CO2', 'H2O']
        fallback_colors = {'SO2': 'tab:red', 'CO2': 'tab:green', 'H2O': 'tab:purple'}
        for sp in guide_species:
            val = input_mix.get(sp, None)
            if val is None:
                continue
            val = float(val)
            if np.isfinite(val) and val > 0.0:
                guide_color = species_colors.get(sp, fallback_colors[sp])
                ax.axvline(
                    val,
                    linestyle='--',
                    linewidth=1.4,
                    color=guide_color,
                    alpha=0.9,
                    label=f'{sp} input',
                )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-10,1.2)
    ax.set_ylim(*ylim)
    ax.legend(ncol=2,bbox_to_anchor=(1.02, 1.02), loc='upper left')

    ax1 = ax.twiny()
    ax1.plot(T, P/1e6, c='k', lw=2, ls='--', label='Temp.')
    ax1.set_xlabel('Temperature (K)')
    ax1.legend(ncol=1,bbox_to_anchor=(1.02, .2), loc='upper left')
    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')

def run(P_surf, mix, verbose=False, model=None):
    """Run the coupled climate equilibrium solve.

    Parameters
    ----------
    P_surf : float
        Surface pressure in dyn/cm^2.
    mix : dict
        Mapping from species name to relative abundance (unitless mole-fraction
        weights). Values are normalized internally by the climate solver.
    verbose : bool, optional
        If ``True``, enables verbose output from the fixed-point solver.
    model : AdiabatClimateEquilibrium or None, optional
        Climate model instance to use. If ``None``, the module-level default
        model is lazily created and reused.

    Returns
    -------
    tuple
        ``(P, T, mix)`` where ``P`` is pressure in dyn/cm^2, ``T`` is
        temperature in K, and ``mix`` maps species names to unitless
        mixing-ratio profiles.
    """

    c = get_climate_model() if model is None else model

    # Solve
    solve_history = c.solve(P_surf, mix, verbose=verbose)

    # Return P, T, mix
    P, T, mix = c.return_atmosphere()

    return P, T, mix

def example():
    """Run a demonstration climate calculation and save a figure.

    Notes
    -----
    Uses ``P_surf = 1.0e6`` dyn/cm^2 (1 bar) and unitless species abundances.
    The generated figure plots pressure in bar (``P/1e6``) and temperature in K.
    """

    P_surf = 1.0e6 # dynes/cm^2
    # Some composition
    mix_input = {
        'CO2': 0.5,
        'H2O': 0.2,
        'SO2': 0.29,
        'H2': 0.01
    }
    P, T, mix = run(P_surf, mix_input, verbose=True)

    plot(
        P,
        T,
        mix,
        ylim=(P[0]/1e6, P[-1]/1e6),
        filename='atmosphere_model/figures/test.pdf',
        P_ref=1e3,
        input_mix=mix_input,
    )

if __name__ == "__main__":
    _ = threadpool_limits(limits=4) # set number of threads
    example()
