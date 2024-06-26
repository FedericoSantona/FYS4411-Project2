# Keywords: vmc, grid, int, boot, plot


if [ -z "$1" ]
then
    python3 src/simulation_scripts/vmc_playground.py
elif [ "$1" == vmc ]
then
    python3 src/simulation_scripts/vmc_playground.py
elif [ "$1" == grid ]
then
    python3 src/simulation_scripts/grid_search.py
elif [ "$1" == int ]
then
    python3 src/simulation_scripts/interaction_vs_particles.py
elif [ "$1" == energy ]
then
     python3 src/simulation_scripts/energy_particle.py
elif [ "$1" == boot ]
then
    python3 src/simulation_scripts/bootstrap.py
elif [ "$1" == plot ]
then 
    python3 src/simulation_scripts/plot_producer.py
elif [ "$1" == deltat ]
then
    python3 src/simulation_scripts/delta_t.py
elif [ "$1" == onebody ]
then
    python3 src/simulation_scripts/one_body.py
elif [ "$1" == reldist ]
then
    python3 src/simulation_scripts/bosons_vs_fermions.py

else
    python3 src/simulation_scripts/"$1"
fi
