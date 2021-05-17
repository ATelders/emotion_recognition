#project_template

The file environment.yml contains the list of all packages needed for the anaconda environment.

To recreate the same environment, type :
    conda env create -f environment.yml

To install a package, type :
    conda install <package>

To save the environment after installing packages, type :
    conda env export > environment.yml
