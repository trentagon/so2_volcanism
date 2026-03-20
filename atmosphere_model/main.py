from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

import os
THISFILE = os.path.dirname(os.path.abspath(__file__))

try:
    from .models import AdiabatClimateEquilibrium, EvoAtmosphereRobust
except ImportError:
    from models import AdiabatClimateEquilibrium, EvoAtmosphereRobust

_CLIMATE_MODEL = None
_PHOTOCHEMICAL_MODEL = None

def _build_default_climate_model():
    return AdiabatClimateEquilibrium(
        species_file=os.path.join(THISFILE, "input/species_climate.yaml"),
        settings_file=os.path.join(THISFILE, "input/settings_climate.yaml"),
        flux_file=os.path.join(THISFILE, "input/gj176_scaled_to_l9859b.txt"),
        thermo_file=os.path.join(THISFILE, "input/thermo.yaml"),
    )

def get_climate_model():
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

def plot_comparison(P_eq, mix_eq, P_photo, T_photo, mix_photo, ylim, filename):
    species_to_plot = ['H2O','CO2','SO2','CO','H2','H2S','S2','S3','S4','S8']

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1,1,figsize=[6,4.5])

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(species_to_plot)))
    for i, sp in enumerate(species_to_plot):
        if sp not in mix_eq or sp not in mix_photo:
            continue
        color = colors[i]
        ax.plot(mix_eq[sp], P_eq/1e6, lw=2, c=color, ls='-', label=sp)
        ax.plot(mix_photo[sp], P_photo/1e6, lw=2, c=color, ls=':')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-10, 1.2)
    ax.set_ylim(*ylim)

    ax1 = ax.twiny()
    ax1.plot(T_photo, P_photo/1e6, c='k', lw=2, ls='--')
    ax1.set_xlabel('Temperature (K)')

    species_legend = ax.legend(ncol=2, bbox_to_anchor=(1.02, 1.02), loc='upper left')
    style_handles = [
        Line2D([0], [0], color='k', lw=2, ls='-', label='Equilibrium'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='Photochemical'),
        Line2D([0], [0], color='k', lw=2, ls='--', label='Temperature'),
    ]
    ax1.legend(handles=style_handles, bbox_to_anchor=(1.02, 0.25), loc='upper left')
    ax.add_artist(species_legend)

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

def run_photochemistry(P, T, mix, Kzz, verbose=True, model=None):
    """Run the photochemical model to steady state from a prescribed atmosphere.

    Parameters
    ----------
    P : ndarray
        Pressure profile in dyn/cm^2.
    T : ndarray
        Temperature profile in K.
    mix : dict
        Mapping from species name to unitless mixing-ratio profiles aligned
        with ``P`` and ``T``.
    Kzz : ndarray
        Eddy diffusion coefficient profile in cm^2/s.
    verbose : bool, optional
        If ``True``, enables verbose output during the robust steady-state
        search.
    model : EvoAtmosphereRobust or None, optional
        Photochemical model instance to use. If ``None``, the module-level
        default model is lazily created and reused.

    Returns
    -------
    tuple
        ``(P, T, mix)`` where ``P`` is pressure in dyn/cm^2, ``T`` is
        temperature in K, and ``mix`` maps species names to unitless
        mixing-ratio profiles after the steady-state solve.

    Notes
    -----
    The default photochemical setup assumes zero-flux boundary conditions for
    all species.
    """

    # Get model
    pc = get_photochemical_model() if model is None else model
    pc.rdat.verbose = verbose

    # Initialize
    pc.initialize_to_PT(P, T, Kzz, mix)

    # Solve
    pc.find_steady_state_robust()

    # Return
    P1, T1, mix1 = pc.return_atmosphere()
    return P1, T1, mix1

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

def example_photochemistry():

    P_surf = 1.0e6 # dynes/cm^2
    # Some composition
    mix = {
        'CO2': 0.5,
        'H2O': 0.2,
        'SO2': 0.29,
        'H2': 0.01
    }

    # Run climate/equilibrium chem
    P, T, mix = run(P_surf, mix, verbose=True)

    # Choose a Kzz vs P, then run photochemistry
    Kzz = np.ones_like(P)*1.0e6
    P1, T1, mix1 = run_photochemistry(P, T, mix, Kzz)

    # Plot comparison
    plot_comparison(
        P,
        mix,
        P1,
        T1,
        mix1,
        ylim=(P1[0]/1e6, P1[-1]/1e6),
        filename='figures/test_photochemistry.pdf',
    )

if __name__ == "__main__":
    _ = threadpool_limits(limits=4) # set number of threads
    example()
    example_photochemistry()
