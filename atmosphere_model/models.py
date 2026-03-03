import numpy as np
from tempfile import NamedTemporaryFile
from copy import deepcopy
import numba as nb
from numba import types
from scipy import integrate
from scipy import constants as const
from scipy import interpolate
import yaml
from copy import deepcopy

try:
    from .fixedpoint import RobustFixedPointSolver
except ImportError:
    from fixedpoint import RobustFixedPointSolver

from photochem import EvoAtmosphere, PhotoException
from photochem.clima import AdiabatClimate, ClimaException
from photochem.equilibrate import ChemEquiAnalysis

### Modified version of Climate model ###

class AdiabatClimateEquilibrium(AdiabatClimate):

    def __init__(self, species_file, settings_file, flux_file, thermo_file=None, data_dir=None):
        """Initialize the climate-equilibrium wrapper.

        Parameters
        ----------
        species_file : str
            Path to the climate species YAML file. The species entries include
            elemental composition (unitless stoichiometric counts).
        settings_file : str
            Path to the climate settings YAML file.
        flux_file : str
            Path to the stellar flux file used by the climate model.
        thermo_file : str or None, optional
            Path to the thermo/species file used by ``ChemEquiAnalysis``.
            If ``None``, ``species_file`` is used.
        data_dir : str or None, optional
            Optional path to a photochem/climate data directory.

        Notes
        -----
        Pressures in this class are handled in cgs units (dyn/cm^2) unless
        explicitly converted when calling equilibrium chemistry helpers.
        """

        super().__init__(
            species_file, 
            settings_file, 
            flux_file,
            data_dir=data_dir
        )

        # Change defaults
        self.P_top = 1.0 # dynes/cm^2
        self.use_make_column_P_guess = False
        self.verbose = False

        if thermo_file is None:
            thermo_file = species_file

        # Save an equilibrium solver
        self.eqsolver = ChemEquiAnalysis(thermo_file)

        # Do some extra work to get composition
        with open(species_file,'r') as f:
            species_dict = yaml.load(f, Loader=yaml.Loader)
        self.species_composition = {}
        for sp in species_dict['species']:
            self.species_composition[sp['name']] = sp['composition']

    def compute_P_grid(self, P_surf):
        """Construct the pressure grid used for equilibrium calculations.

        Parameters
        ----------
        P_surf : float
            Surface pressure in dyn/cm^2.

        Returns
        -------
        ndarray
            1D pressure grid in dyn/cm^2, including the surface level and
            extending to ``self.P_top``.
        """
        P_top =self.P_top
        nz = len(self.T)
        P_grid = np.logspace(np.log10(P_surf), np.log10(P_top), 2*nz+1)
        P_grid = np.append(P_grid[0], P_grid[1::2])
        return P_grid

    def RCE_robust(self, P_i, T_init, custom_dry_mix):
        """Run radiative-convective equilibrium with fallback temperature guesses.

        Parameters
        ----------
        P_i : ndarray
            Surface partial pressures for model species in dyn/cm^2. Ordering
            must match ``self.species_names``.
        T_init : ndarray
            Initial temperature state in K. Expected shape is ``(nz + 1,)``
            with ``T_init[0]`` as surface temperature and the remainder as
            atmospheric layer temperatures.
        custom_dry_mix : dict
            Dry composition profiles passed through to ``self.RCE``. Must
            include key ``'pressure'`` in dyn/cm^2 and per-species mixing-ratio
            profiles (unitless mole fractions).

        Returns
        -------
        bool
            ``True`` if an RCE solution converged, otherwise ``False``.
        """

        T_surf_guess = T_init[0]
        T_guess = T_init[1:]
        try:
            converged = self.RCE(P_i, T_surf_guess, T_guess, custom_dry_mix=custom_dry_mix)
        except ClimaException:
            converged = False

        if converged:
            return True

        T_guess_mid = self.rad.equilibrium_temperature(0.0)*1.5
        T_perturbs = np.array([0.0, 50.0, -50.0, 100.0, -100.0, 150.0, 800.0, 600.0, 400.0, 300.0, 200.0])

        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            T_guess = np.ones(self.T.shape[0])*T_surf_guess
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess, custom_dry_mix=custom_dry_mix)
                if converged:
                    break
            except ClimaException:
                converged = False

        return converged
    
    def get_molfracs_atoms(self, mix):
        """Convert species mole fractions into elemental mole fractions.

        Parameters
        ----------
        mix : dict
            Mapping from species name to species mole fraction (unitless) at
            the reference level used to define elemental abundances.

        Returns
        -------
        ndarray
            Elemental mole fractions (unitless), ordered as
            ``self.eqsolver.atoms_names`` and normalized to sum to 1.
        """

        molfracs_atoms_sun = np.ones(len(self.eqsolver.atoms_names))*1e-10
        for sp in mix:
            for i,atom in enumerate(self.eqsolver.atoms_names):
                if atom in self.species_composition[sp]:
                    molfracs_atoms_sun[i] += self.species_composition[sp][atom]*mix[sp]
        molfracs_atoms_sun /= np.sum(molfracs_atoms_sun)

        return molfracs_atoms_sun

    def g_eval(self, T, P_surf, mix):
        """Evaluate the fixed-point map for climate-chemistry coupling.

        Parameters
        ----------
        T : ndarray
            Temperature state in K for the full column, shape ``(nz + 1,)``.
            Entry 0 is surface temperature; remaining entries are atmospheric
            temperatures.
        P_surf : float
            Surface pressure in dyn/cm^2.
        mix : dict
            Species mole fractions (unitless) used to compute elemental
            abundances for equilibrium chemistry.

        Returns
        -------
        ndarray
            Updated temperature state in K with shape ``(nz + 1,)``. Returns
            ``NaN`` values if the internal climate solve does not converge.
        """

        eqsolver = self.eqsolver

        # Compute the P grid
        P_grid = self.compute_P_grid(P_surf)

        # Compute chemical equilibrium
        molfracs_atoms = self.get_molfracs_atoms(mix)
        gases, condensates = equilibrate_atmosphere(eqsolver, P_grid/1e6, T, molfracs_atoms)
        # Copy to climate model
        P_i = np.ones(len(self.species_names))*1.0e-30
        custom_dry_mix = {'pressure': P_grid}
        for i,sp in enumerate(self.species_names):
            custom_dry_mix[sp] = np.maximum(gases[sp],1.0e-30)
            P_i[i] = np.maximum(gases[sp][0], 1.0e-30)*P_grid[0]
        assert np.isclose(np.sum(P_i), P_grid[0])

        # Solve climate
        converged = self.RCE_robust(P_i, T, custom_dry_mix)
        if not converged:
            return np.zeros_like(T)*np.nan
        
        return np.append(self.T_surf, self.T)
    
    def solve(self, P_surf, mix, *, tol=1.0, max_tol=2.0, **kwargs):
        """Solve the climate fixed-point problem for a given surface pressure.

        Parameters
        ----------
        P_surf : float
            Surface pressure in dyn/cm^2.
        mix : dict
            Species mole fractions (unitless). Values are normalized internally
            before use.
        tol : float, optional
            Convergence tolerance for the scaled RMS fixed-point residual in K.
        max_tol : float, optional
            Maximum allowed per-component scaled residual in K.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`fixedpoint.RobustFixedPointSolver`.

        Returns
        -------
        SolveResult
            Result object returned by
            :class:`fixedpoint.RobustFixedPointSolver.solve`.
        """

        # Normalize
        mix_ = deepcopy(mix)
        f_tot = 0.0
        for sp in mix_:
            val = float(mix_[sp])
            if not np.isfinite(val):
                raise ValueError(f"mix[{sp!r}] must be finite; got {mix_[sp]!r}")
            if val < 0.0:
                raise ValueError(f"mix[{sp!r}] must be >= 0; got {mix_[sp]!r}")
            f_tot += val
        if f_tot <= 0.0:
            raise ValueError("Sum of mix values must be > 0 for normalization.")
        for sp in mix:
            mix_[sp] /= f_tot

        def g(x):
            return self.g_eval(x, P_surf, mix_)

        x0 = np.ones(len(self.T)+1)*self.rad.equilibrium_temperature(0.0)*1.5
        solver = RobustFixedPointSolver(
            g=g,
            x0=x0,
            tol=tol,
            max_tol=max_tol,
            **kwargs
        )
        result = solver.solve()

        if not result.converged:
            msg = (
                "AdiabatClimateEquilibrium fixed-point solve failed to converge "
                f"(iters={result.iters}, func_evals={result.func_evals})."
            )
            if result.history:
                k, _, r_k, rnorm, omega_k, beta_k = result.history[-1]
                rmax = float(np.max(np.abs(r_k)))
                msg += (
                    f" Last iter={k}, residual_rms={rnorm:.3e}, "
                    f"residual_max_abs={rmax:.3e}, omega={omega_k:.3f}, beta={beta_k:.3f}."
                )
            raise RuntimeError(msg)

        return result
    
    def return_atmosphere(self):
        """Return the current climate state.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            ``(P, T, mix)`` where ``P`` is pressure in dyn/cm^2 (surface
            included), ``T`` is temperature in K (surface included), and
            ``mix`` maps species names to unitless mixing-ratio profiles.
        """
        f_i = np.concatenate((self.f_i_surf.reshape((1,len(self.f_i_surf))),self.f_i))
        P = np.append(self.P_surf, self.P)
        T = np.append(self.T_surf, self.T)

        mix = {}
        for i,sp in enumerate(self.species_names):
            mix[sp] = f_i[:,i]

        return P, T, mix

def equilibrate_atmosphere(eqsolver, P, T, molfracs_atoms):

    # Check inputs
    if not isinstance(P, np.ndarray):
        raise ValueError('`P` should be a numpy array.')
    if not isinstance(T, np.ndarray):
        raise ValueError('`T` should be a numpy array.')

    # Some conversions and copies
    P_cgs = P*1e6
    gas_names = eqsolver.gas_names
    condensate_names = eqsolver.condensate_names

    gases = {}
    for key in gas_names:
        gases[key] = np.empty(len(P))
    condensates = {}
    for key in condensate_names:
        condensates[key] = np.empty(len(P))

    for i in range(len(P)):
        if i > 0:
            eqsolver.use_prev_guess = True
        # Try many perturbations on T to try to get convergence
        for eps in [0.0, 1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-4, -1.0e-4]:
            converged = eqsolver.solve(P_cgs[i], T[i] + T[i]*eps, molfracs_atoms=molfracs_atoms)
            if converged:
                break
        if not converged:
            # We will not enforce convergence.
            pass
        molfracs_species_gas = eqsolver.molfracs_species_gas
        molfracs_species_condensate = eqsolver.molfracs_species_condensate
        for j,key in enumerate(gas_names):
            gases[key][i] = molfracs_species_gas[j]
        for j,key in enumerate(condensate_names):
            condensates[key][i] = molfracs_species_condensate[j]
    eqsolver.use_prev_guess = False

    return gases, condensates

### Modified version of Photochemical model ###

class RobustData():
    
    def __init__(self):

        # Parameters for determining steady state
        self.atols = [1e-23, 1e-22, 1e-20, 1e-18]
        self.min_mix_reset = -1e-13
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 3 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.freq_update_atol = 10_000
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = True # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.max_time = None
        self.robust_stepper_initialized = None
        # Surface pressures
        self.Pi = None

class EvoAtmosphereRobust(EvoAtmosphere):
    """Photochemical model wrapper with robust initialization and stepping.

    This class extends :class:`photochem.EvoAtmosphere` with helpers to:

    - initialize from climate-model ``P-T-Kzz`` and composition profiles,
    - apply and restore surface-pressure boundary conditions,
    - integrate with reinitialization safeguards and adaptive tolerances,
    - save/restore full model state during robust solves.
    """

    def __init__(self, mechanism_file, settings_file, flux_file, data_dir=None):
        """Initialize the robust photochemical model.

        Parameters
        ----------
        mechanism_file : str
            Path to reaction-mechanism YAML file.
        settings_file : str
            Path to photochemical settings YAML file.
        flux_file : str
            Path to stellar flux file.
        data_dir : str, optional
            Optional photochem data directory.
        """

        with NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(ATMOSPHERE_INIT)
            f.flush()
            super().__init__(
                mechanism_file, 
                settings_file, 
                flux_file,
                f.name,
                data_dir
            )

        self.rdat = RobustData()

        # Values in photochem to adjust
        self.var.verbose = 0
        self.var.upwind_molec_diff = True
        self.var.autodiff = True
        self.var.atol = 1.0e-23
        self.var.equilibrium_time = 1e15
        self.var.conv_longdy = 1e-3

        # Model state
        self.max_time_state = None

        for i in range(len(self.var.cond_params)):
            self.var.cond_params[i].smooth_factor = 1
            self.var.cond_params[i].k_evap = 0

    def set_surface_pressures(self, Pi):
        """Set lower boundary pressures for selected species.

        Parameters
        ----------
        Pi : dict
            Mapping ``{species_name: partial_pressure_dyn_cm2}``.
        """
        
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

    def initialize_to_PT(self, P, T, Kzz, mix):
        """Initialize model state from target pressure-temperature-composition data.

        Parameters
        ----------
        P : ndarray
            Pressure profile in dynes/cm^2 (surface to top).
        T : ndarray
            Temperature profile in K on ``P``.
        Kzz : ndarray
            Eddy diffusion profile in cm^2/s on ``P``.
        mix : dict
            Mapping of species name to mixing-ratio profile on ``P``.
        """

        P, T, mix = deepcopy(P), deepcopy(T), deepcopy(mix)

        rdat = self.rdat

        # Ensure X sums to 1
        ftot = np.zeros(P.shape[0])
        for key in mix:
            ftot += mix[key]
        for key in mix:
            mix[key] = mix[key]/ftot

        # Compute mubar at all heights
        mu = {}
        for i,sp in enumerate(self.dat.species_names[:-2]):
            mu[sp] = self.dat.species_mass[i]
        mubar = np.zeros(P.shape[0])
        for key in mix:
            mubar += mix[key]*mu[key]

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P, T, mubar, self.dat.planet_radius, self.dat.planet_mass, rdat.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
        if P1.shape[0] != Kzz.shape[0]:
            Kzz1 = np.append(Kzz,Kzz[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz.copy()
            mix1 = mix

        rdat.log10P_interp = np.log10(P1.copy()[::-1])
        rdat.T_interp = T1.copy()[::-1]
        rdat.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        
        # extrapolate to 1e6 bars
        T_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.T_interp, bounds_error=False, fill_value='extrapolate')(12)
        edd_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.log10edd_interp, bounds_error=False, fill_value='extrapolate')(12)
        rdat.log10P_interp = np.append(rdat.log10P_interp, 12)
        rdat.T_interp = np.append(rdat.T_interp, T_tmp)
        rdat.log10edd_interp = np.append(rdat.log10edd_interp, edd_tmp)

        rdat.P_desired = P1.copy()
        rdat.T_desired = T1.copy()
        rdat.Kzz_desired = Kzz1.copy()

        # Calculate the photochemical grid
        ind_t = np.argmin(np.abs(P1 - rdat.TOA_pressure_avg))
        z_top = z1[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1, np.log10(P1))
        T_p = np.interp(z_p, z1, T1)
        Kzz_p = 10.0**np.interp(z_p, z1, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Update photochemical model grid
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        self.prep_atmosphere(self.wrk.usol)

    def initialize_to_PT_bcs(self, P, T, Kzz, mix, Pi):
        """Initialize from ``P-T-Kzz-mix`` and set surface-pressure BCs.

        Parameters
        ----------
        P : ndarray
            Pressure profile in dynes/cm^2.
        T : ndarray
            Temperature profile in K.
        Kzz : ndarray
            Eddy diffusion profile in cm^2/s.
        mix : dict
            Species mixing-ratio profiles.
        Pi : dict
            Surface partial-pressure boundary conditions in dynes/cm^2.
        """
        self.rdat.Pi = Pi
        self.set_surface_pressures(Pi)
        self.initialize_to_PT(P, T, Kzz, mix)

    def set_particle_radii(self, radii):
        """Set particle radii profiles for selected species.

        Parameters
        ----------
        radii : dict
            Mapping ``{species_name: radius_profile_cm}``.
        """
        particle_radius = self.var.particle_radius
        for key in radii:
            ind = self.dat.species_names.index(key)
            particle_radius[ind,:] = radii[key]
        self.var.particle_radius = particle_radius
        self.update_vertical_grid(TOA_alt=self.var.top_atmos)

    def initialize_robust_stepper(self, usol):
        """Initialize the robust integrator state.

        Parameters
        ----------
        usol : ndarray
            Number-density state array.
        """
        rdat = self.rdat  
        rdat.total_step_counter = 0
        rdat.nerrors = 0
        rdat.max_time = 0
        self.max_time_state = None
        self.initialize_stepper(usol)
        rdat.robust_stepper_initialized = True

    def robust_step(self):
        """Take one safeguarded integration step.

        Returns
        -------
        tuple
            ``(give_up, reached_steady_state)``.
        """

        rdat = self.rdat

        if not rdat.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                rdat.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Reset integrator if we get large magnitude negative numbers
            if not self.healthy_atmosphere():
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Update the max time achieved
            if self.wrk.tn > rdat.max_time:
                rdat.max_time = self.wrk.tn
                self.max_time_state = self.model_state_to_dict() # save the model state

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > rdat.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < rdat.max_dT_tol and max_dlog10edd < rdat.max_dlog10edd_tol and rdat.TOA_pressure_avg/3 < TOA_pressure < rdat.TOA_pressure_avg*3

            if condition1 and condition2:
                if rdat.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (rdat.total_step_counter % rdat.freq_update_atol):
                ind = int(rdat.total_step_counter/rdat.freq_update_atol)
                ind1 = ind - len(rdat.atols)*int(ind/len(rdat.atols))
                self.var.atol = rdat.atols[ind1]
                if rdat.verbose:
                    print('new atol = %.1e'%(self.var.atol))
                self.initialize_stepper(self.wrk.usol)
                break

            if not (self.wrk.nsteps % rdat.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid, if possible.
                try:
                    self.set_press_temp_edd(rdat.P_desired,rdat.T_desired,rdat.Kzz_desired,hydro_pressure=True)
                except PhotoException:
                    pass
                try:
                    self.update_vertical_grid(TOA_pressure=rdat.TOA_pressure_avg)
                except PhotoException:
                    pass
                self.initialize_stepper(self.wrk.usol)

            if rdat.total_step_counter > rdat.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % rdat.freq_print) and rdat.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Integrate until steady state or a stop condition is reached.

        Returns
        -------
        bool
            ``True`` if steady-state convergence is reached, else ``False``.
        """

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def healthy_atmosphere(self):
        """Check for unphysical negative mixing-ratio excursions.

        Returns
        -------
        bool
            ``True`` if atmosphere history satisfies the configured lower bound.
        """
        return np.min(self.wrk.mix_history[:,:,0]) > self.rdat.min_mix_reset
    
    def find_steady_state_robust(self):
        """Try multiple tolerance settings to recover steady-state convergence.

        Returns
        -------
        bool
            ``True`` if any robust attempt converges, else ``False``.
        """

        # Change some rdat settings
        self.rdat.freq_update_atol = 100_000
        self.rdat.max_total_step = 10_000

        # First just try to get to steady-state with standard atol
        self.var.atol = 1.0e-23
        converged = self.find_steady_state()
        if converged:
            return converged

        # Convergence did not happen. Save the max time state.
        max_time = self.rdat.max_time
        max_time_state = deepcopy(self.max_time_state)

        # Lets try a couple different atols.
        for atol in [1.0e-18, 1.0e-15]:
            # Lets initialize to max time state
            self.initialize_from_dict(max_time_state)
            # Do some smaller number of steps
            self.rdat.max_total_step = 5_000
            self.var.atol = atol # set the atol
            converged = self.find_steady_state() # Integrate
            if converged:
                # If converged then lets return
                return converged

            # No convergence. We re-save max time state
            if self.rdat.max_time > max_time:
                max_time = self.rdat.max_time
                max_time_state = deepcopy(self.max_time_state)

        # No convergence, we reinitialize to max time state and return
        self.initialize_from_dict(max_time_state)

        return converged
    
    def return_atmosphere(self):
        """Return current photochemical atmosphere fields.

        Returns
        -------
        tuple
            ``(P, T, mix)`` where ``P`` is pressure (dynes/cm^2), ``T`` is
            temperature (K), and ``mix`` is a species->mixing-ratio dict.
        """

        T = self.var.temperature
        P = self.wrk.pressure_hydro
        mix = self.mole_fraction_dict()
        mix.pop('alt')
        mix.pop('pressure')
        mix.pop('density')
        mix.pop('temp')

        return P, T, mix

    def model_state_to_dict(self):
        """Serialize model state needed for restart.

        Returns
        -------
        dict
            Restart dictionary compatible with :meth:`initialize_from_dict`.
        """

        if self.rdat.log10P_interp is None:
            raise Exception('This routine can only be called after `initialize_to_PT_bcs`')

        out = {}
        out['rdat'] = deepcopy(self.rdat.__dict__)
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['particle_radius'] = self.var.particle_radius

        # Other settings
        out['equilibrium_time'] = self.var.equilibrium_time
        out['verbose'] = self.var.verbose
        out['atol'] = self.var.atol
        out['autodiff'] = self.var.autodiff

        return out

    def initialize_from_dict(self, out):
        """Restore model state from :meth:`model_state_to_dict` output.

        Parameters
        ----------
        out : dict
            Restart dictionary created by :meth:`model_state_to_dict`.
        """

        for key, value in out['rdat'].items():
            setattr(self.rdat, key, value)

        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']
        self.var.particle_radius = out['particle_radius']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])

        # Other settings
        self.var.equilibrium_time = out['equilibrium_time']
        self.var.verbose = out['verbose']
        self.var.atol = out['atol']
        self.var.autodiff = out['autodiff']
        
        # Now set boundary conditions
        Pi = self.rdat.Pi
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

        self.prep_atmosphere(self.wrk.usol)

@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:] # type: ignore
    T : types.double[:] # type: ignore
    mubar : types.double[:] # type: ignore

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    if P_top < P[-1]:
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Integrate to TOA
    out = integrate.solve_ivp(hydrostatic_equation, [P_[0], P_[-1]], np.array([0.0]), t_eval=P_, args=args, rtol=1e-6)
    assert out.success

    # Stitch together
    z_ = out.y[0]

    return P_, T_, mubar_, z_

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""
