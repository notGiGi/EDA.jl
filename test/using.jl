using Pkg
Pkg.add(url="https://github.com/GiGi/GiGi.git")
using GiGi  # Importa tu módulo EDA
using DataFrames
using Plots
# Datos de prueba
data = DataFrame(
    col1 = [1, 2, 3, 4, 5],
    col2 = [2, 4, 6, 8, 10],
    col3 = [5, 4, 3, 2, 1]
)

loader = EDALoader(data)

