using Test
using EDA  # Importa tu módulo EDA
using DataFrames
using Plots
# Datos de prueba
data = DataFrame(
    col1 = [1, 2, 3, 4, 5],
    col2 = [2, 4, 6, 8, 10],
    col3 = [5, 4, 3, 2, 1]
)

loader = EDALoader(data)  # Carga inicial del DataFrame

# Test para la función visualize_data
@testset "Visualize Data" begin
    preview = visualize_data(loader, n=3)
    @test nrow(preview) == 3  # Debe mostrar solo 3 filas
end

# Test para la función correlation
@testset "Correlation" begin
    corr_matrix = correlation(loader)
    @test size(corr_matrix) == (3, 3)  # Debe ser una matriz 3x3
    @test corr_matrix[1, 2] ≈ 1.0  # col1 y col2 tienen correlación perfecta
end

# Test para la función threshold
@testset "Threshold for Missing Data" begin
    data_with_missing = DataFrame(
        col1 = [1, missing, 3, missing, 5],  # 40% de valores faltantes
        col2 = [2, 4, 6, 8, 10]              # 0% de valores faltantes
    )
    loader_with_missing = EDALoader(data_with_missing)
    
    # Aplicar threshold con un umbral de 30%
    filtered_data = threshold(loader_with_missing, 30.0)  # Elimina columnas con >30% faltantes
    
    # Verificar que solo quede col2
    @test ncol(filtered_data) == 1  # Solo queda col2
    @test names(filtered_data) == ["col2"]  # Asegura que col2 sea la única columna
end



