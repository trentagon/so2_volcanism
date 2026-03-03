from photochem.utils import settings_file_for_climate, species_file_for_climate, stars
from photochem.utils._format import yaml, FormatSettings_main, MyDumper
from photochem.utils import zahnle_rx_and_thermo_files
from astropy import constants
import os
THISFILE = os.path.dirname(os.path.abspath(__file__))
try:
    from . import planets
except ImportError:
    import planets

def main():

    planet_mass = float(constants.M_earth.cgs.value)*planets.L9859b.mass
    planet_radius = float(constants.R_earth.cgs.value)*planets.L9859b.radius
    surface_albedo = 0.1

    zahnle_rx_and_thermo_files(
        atoms_names=['H','O','C','S'],
        rxns_filename=os.path.join(THISFILE, 'input/zahnle_HOCS.yaml'),
        thermo_filename=os.path.join(THISFILE, 'input/thermo.yaml')
    )
    with open(os.path.join(THISFILE, 'input/zahnle_HOCS.yaml'),'r') as f:
        dat = yaml.load(f, yaml.Loader)
    species = []
    for i,sp in enumerate(dat['species']):
        species.append(sp['name'])

    species_file_for_climate(
        filename=os.path.join(THISFILE, 'input/species_climate.yaml'),
        species=species,
        condensates=[]
    )

    settings_file_for_climate(
        filename=os.path.join(THISFILE, 'input/settings_climate.yaml'),
        planet_mass=planet_mass,
        planet_radius=planet_radius,
        surface_albedo=surface_albedo
    )

    stars.muscles_spectrum(
        star_name='GJ176',
        outputfile=os.path.join(THISFILE, 'input/gj176_scaled_to_l9859b.txt'),
        Teq=planets.L9859b.Teq
    )

    settings_file = {
        'atmosphere-grid': {
            'bottom': 0.0, 
            'top': 'atmospherefile', 
            'number-of-layers': 100
        },
        'planet': {
            'planet-mass': planet_mass,
            'planet-radius': planet_radius,
            'surface-albedo': surface_albedo,
            'solar-zenith-angle': 60.0,
            'hydrogen-escape': {'type': 'none'},
            'water': {'fix-water-in-troposphere': False, 'gas-rainout': False, 'water-condensation': False}
        },
        'boundary-conditions': [{
            'name': 'H2',
            'lower-boundary': {'type': 'vdep', 'vdep': 0.0},
            'upper-boundary': {'type': 'veff', 'veff': 0.0}
        }]
    }
    settings_file = FormatSettings_main(settings_file)
    with open(os.path.join(THISFILE, 'input/settings.yaml'),'w') as f:
        yaml.dump(settings_file,f,Dumper=MyDumper,sort_keys=False,width=70)

if __name__ == '__main__':
    main()
