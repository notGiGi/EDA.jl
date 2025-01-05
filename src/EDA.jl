module GiGi
export EDALoader, load, visualize_data, barstackmissing, describedata, dataType, threshold, correlation, heat, outlierswithiqr, outlierhandle, deleteRow, filterColumnsByCorrelation, correlation_network, linear_regression, save_to_csv, linearregression

using CSV, DataFrames, VegaLite, Statistics, Missings, StatsBase, ColorSchemes, Plots, Graphs, GraphPlot, Colors, Flux, CategoricalArrays
using Optimisers, GLM, Plots, DataFrames, Statistics, CSV



"""
Defines a structure to hold the loaded DataFrame and its associated cache.
This structure allows easy manipulation of data while maintaining consistency
after operations are applied.

Example:
    loader = EDALoader(df)
"""
# Estructura que encapsula el DataFrame y operaciones relacionadas
mutable struct EDALoader
    data::DataFrame
    cache::Dict{Symbol, Any}
end

# Constructor de EDALoader
function EDALoader(data::DataFrame)
    cache = Dict{Symbol, Any}()
    loader = EDALoader(data, cache)
    update_cache!(loader)
    return loader
end


"""
Loads a dataset from a file and prepares it for analysis. The `na` parameter
specifies how missing values are represented in the CSV.

Args:
    file::String: Path to the CSV file.
    na::Any: Representation of missing values in the file.

Returns:
    EDALoader: An instance encapsulating the loaded data.

Example:
    data = load("path/to/file.csv", "NA")
"""

# Método para cargar datos desde un archivo
function load(file::String, na::Any)
    if !isfile(file)
        throw(ArgumentError("El archivo especificado no existe: $file"))
    end
    data = DataFrame(CSV.File(file, missingstring=[na]))
    return EDALoader(data)
end


"""
Updates the cache dynamically whenever changes are made to the DataFrame.
This function ensures that cached metadata (e.g., column names, types,
missing value counts) is always up to date.
"""

# Método para actualizar el cache dinámicamente
function update_cache!(loader::EDALoader)
    data = loader.data
    loader.cache[:n] = nrow(data)
    loader.cache[:namecol] = names(data)
    loader.cache[:typecol] = [eltype(col) for col in eachcol(data)]
    loader.cache[:numberm] = [count(ismissing, col) for col in eachcol(data)]
    loader.cache[:missingdataframe] = DataFrame(
        "Name" => loader.cache[:namecol],
        "Type of the column" => loader.cache[:typecol],
        "Numbermissing" => loader.cache[:numberm]
    )
end

"""
Provides a preview of the data, showing the first `n` rows.

Args:
    loader::EDALoader: The data loader object.
    n::Int: Number of rows to preview (default: 10).

Returns:
    DataFrame: A preview of the data.

Example:
    preview = visualize_data(loader, n=5)
"""


# Método para visualizar el DataFrame
function visualize_data(loader::EDALoader; n::Int = 10)
    return first(loader.data, n)
end

"""
Generates a summary of the DataFrame, including the number of rows and columns.

Args:
    loader::EDALoader: The data loader object.

Returns:
    DataFrame: A summary with the number of rows and columns.

Example:
    summary = describedata(loader)
"""


# Describir los datos
function describedata(loader::EDALoader)
    update_cache!(loader)
    return DataFrame(
        "Number of rows" => nrow(loader.data),
        "Number of columns" => ncol(loader.data)
    )
end

"""
Displays the data types of all columns in the DataFrame.

Args:
    loader::EDALoader: The data loader object.

Returns:
    DataFrame: A summary of column names and their respective data types.

Example:
    types = dataType(loader)
"""



# Tipos de datos
function dataType(loader::EDALoader)
    update_cache!(loader)
    return DataFrame(
        "Name" => names(loader.data),
        "Type of variable" => [eltype(col) for col in eachcol(loader.data)]
    )
end

"""
Creates a bar chart showing the proportion of missing and non-missing data
for each column in the DataFrame.

Args:
    loader::EDALoader: The data loader object.

Returns:
    VegaLite.Plot: A bar chart representing the missing data.

Example:
    barstackmissing(loader)
"""



# Gráfica de datos faltantes
function barstackmissing(loader::EDALoader)
    update_cache!(loader)
    n = loader.cache[:n]
    numberm = loader.cache[:numberm]
    namecol = loader.cache[:namecol]
    percentage = round.((100 / n) .* numberm, digits=1)
    datatot = [n - miss for miss in numberm]
    totdata = DataFrame(
        "Name" => namecol,
        "datatotal" => datatot,
        "Numbermissing" => numberm,
        "total" => n,
        "Percentage of missing data" => percentage
    )
    println(totdata)
    return totdata |> @vlplot(
        :bar,
        transform=[{fold = ["datatotal", "Numbermissing"]}],
        encoding={
            y={field="Name", type="nominal", title="Columns"},
            x={field="value", type="quantitative", stack=:normalize, title="Proportion"},
            color={field="key", type="nominal", title="Data Type"}
        }
    )
end

"""
Filters columns based on a specified missing data threshold. Columns with a
percentage of missing values exceeding the threshold are removed, unless
specified in the `keep_columns` parameter.

Args:
    loader::EDALoader: The data loader object.
    threshold::Float64: The missing value percentage threshold.
    keep_columns::Vector{String}: Columns to keep regardless of the threshold.

Returns:
    DataFrame: The filtered DataFrame.

Example:
    data = threshold(loader, 10.0, keep_columns=["important_column"])
"""


# Filtrar datos por umbral de faltantes
function threshold(loader::EDALoader, threshold::Float64; keep_columns::Vector{String} = String[])
    # Asegurar que el caché esté actualizado
    update_cache!(loader)

    # Calcular el porcentaje de valores faltantes por columna
    totdata = loader.cache[:missingdataframe]
    if :Percentage_of_missing_data ∉ names(totdata)
        totdata[!, :Percentage_of_missing_data] = (totdata[!, :Numbermissing] ./ loader.cache[:n]) .* 100
    end

    # Seleccionar columnas que cumplen con el umbral
    mask = totdata[!, :Percentage_of_missing_data] .<= threshold
    columns_to_keep = vcat(totdata[mask, :Name], keep_columns) |> unique

    # Filtrar el DataFrame según las columnas seleccionadas
    loader.data = select(loader.data, intersect(names(loader.data), columns_to_keep))
    
    # Actualizar el caché después de filtrar
    update_cache!(loader)
    return loader.data
end


"""
Computes the correlation matrix for all numeric columns in the DataFrame.

Args:
    loader::EDALoader: The data loader object.

Returns:
    Matrix{Float64}: The correlation matrix.

Example:
    corr_matrix = correlation(loader)
"""


# Matriz de correlación
function correlation(loader::EDALoader)
    update_cache!(loader)

    # Seleccionar columnas numéricas
    numeric_cols = findall(col -> all(v -> v isa Union{Missing, Number}, col), eachcol(loader.data))
    numeric_data = select(loader.data, numeric_cols)

    # Eliminar columnas con valores faltantes o constantes
    valid_cols = [col for col in eachcol(numeric_data) if count(!ismissing, col) > 1 && std(skipmissing(col)) > 0]
    if isempty(valid_cols)
        throw(ArgumentError("No hay suficientes columnas válidas para calcular la matriz de correlación."))
    end
    numeric_data = hcat(valid_cols...)

    # Calcular la matriz de correlación
    return pairwise(cor, eachcol(numeric_data), skipmissing=:pairwise, symmetric=true)
end



"""
Generates a heatmap to visualize the correlation matrix of numeric columns.

Args:
    loader::EDALoader: The data loader object.

Returns:
    Nothing: Displays the heatmap plot.

Example:
    heat(loader)
"""

# Heatmap de correlaciones
function heat(loader::EDALoader)
    update_cache!(loader)

    # Seleccionar columnas numéricas
    numeric_cols = findall(col -> all(v -> v isa Union{Missing, Number}, col), eachcol(loader.data))
    numeric_data = select(loader.data, numeric_cols)

    # Eliminar columnas con valores faltantes o constantes
    valid_cols = [col for col in eachcol(numeric_data) if count(!ismissing, col) > 1 && std(skipmissing(col)) > 0]
    valid_names = names(numeric_data)[[i for (i, col) in enumerate(eachcol(numeric_data)) if count(!ismissing, col) > 1 && std(skipmissing(col)) > 0]]
    if isempty(valid_cols)
        throw(ArgumentError("No hay suficientes columnas válidas para calcular el heatmap."))
    end
    numeric_data = hcat(valid_cols...)

    # Calcular la matriz de correlación
    corr_matrix = pairwise(cor, eachcol(numeric_data), skipmissing=:pairwise, symmetric=true)

    # Crear el heatmap
    heatmap(
        corr_matrix,
        xticks=(1:size(corr_matrix, 2), valid_names), xrot=90,
        yticks=(1:size(corr_matrix, 2), valid_names), yflip=true, size=(800, 800),
        c=cgrad([:white, :blue, :red])
    )
    annotate!([(j, i, text(round(corr_matrix[i, j], digits=2), 8, "Computer Modern", :black)) for i in 1:size(corr_matrix, 1) for j in 1:size(corr_matrix, 2)])
end



"""
Detects outliers in numeric columns using the Interquartile Range (IQR) method.
Displays quartiles, IQR, and the number of outliers in each column.

Args:
    loader::EDALoader: The data loader object.

Returns:
    DataFrame: A summary of IQR, quartiles, and outliers for each numeric column.

Example:
    outliers = outlierswithiqr(loader)
"""


# Detección de valores atípicos con IQR
function outlierswithiqr(loader::EDALoader)
    update_cache!(loader)
    numeric_data = select(loader.data, findall(col -> all(v -> v isa Union{Missing, Number}, col), eachcol(loader.data)))
    cols = names(numeric_data)
    riq = [iqr(skipmissing(col)) for col in eachcol(numeric_data)]
    q1q = [quantile(skipmissing(col), 0.25) for col in eachcol(numeric_data)]
    q3q = [quantile(skipmissing(col), 0.75) for col in eachcol(numeric_data)]
    lb = q1q .- 1.5 .* riq
    up = q3q .+ 1.5 .* riq
    tc = [count(x -> x < l || x > u, skipmissing(col)) for (col, l, u) in zip(eachcol(numeric_data), lb, up)]
    return DataFrame(
        "Column" => cols,
        "IQR" => riq,
        "Q1" => q1q,
        "Q3" => q3q,
        "Lower Bound IQR" => lb,
        "Upper Bound IQR" => up,
        "Number of outliers" => tc
    )
end

"""
Handles outliers in numeric columns based on the IQR method. Currently, the
only available method is removing rows with outliers.

Args:
    loader::EDALoader: The data loader object.
    option::String: Method for handling outliers (default: "RemoveRow").
    multiplier::Float64: Multiplier for the IQR to define bounds (default: 1.5).

Returns:
    DataFrame: The DataFrame with outliers handled.

Example:
    data_cleaned = outlierhandle(loader, "RemoveRow", multiplier=1.5)
"""


# Manejo de valores atípicos
function outlierhandle(loader::EDALoader, option::String, multiplier::Float64 = 1.5)
    update_cache!(loader)

    # Seleccionar solo columnas numéricas (o con missing)
    numeric_cols = findall(col -> all(v -> v isa Union{Missing, Number}, col), eachcol(loader.data))
    numeric_data = select(loader.data, numeric_cols)

    # Calcular IQR y límites
    riq = [iqr(skipmissing(col)) for col in eachcol(numeric_data)]
    q1q = [quantile(skipmissing(col), 0.25) for col in eachcol(numeric_data)]
    q3q = [quantile(skipmissing(col), 0.75) for col in eachcol(numeric_data)]
    lb = q1q .- multiplier .* riq
    up = q3q .+ multiplier .* riq

    if option == "RemoveRow"
        # Crear máscara para identificar filas válidas
        valid_rows = map(row -> all(i -> begin
            val = row[numeric_cols[i]]
            if val isa Missing
                true  # Ignorar valores missing
            elseif val isa Number
                lb[i] <= val <= up[i]
            else
                false  # Ignorar si no es numérico
            end
        end, 1:length(lb)), eachrow(loader.data))

        # Filtrar filas válidas
        loader.data = loader.data[valid_rows, :]
        update_cache!(loader)
        return loader.data
    end

    return loader.data
end


"""
Deletes rows with missing values in the specified column.

Args:
    loader::EDALoader: The data loader object.
    column::String: Name of the column to check for missing values.

Returns:
    DataFrame: The DataFrame with rows removed.

Example:
    data_no_missing = deleteRow(loader, "column_name")
"""




# Eliminar filas con valores faltantes
function deleteRow(loader::EDALoader, column::String)
    update_cache!(loader)
    if column ∉ names(loader.data)
        throw(ArgumentError("La columna $column no existe en el DataFrame."))
    end
    # Filtrar el DataFrame completo en lugar de asignar un vector
    loader.data = filter(row -> !ismissing(row[column]), loader.data)
    update_cache!(loader)
    return loader.data
end


"""
Filters columns based on their correlation with a target column. Keeps or removes
columns depending on the correlation threshold.

Args:
    loader::EDALoader: The data loader object.
    target::String: Name of the target column.
    threshold::Float64: Correlation threshold for filtering columns.
    relation::Bool: If `true`, keeps columns with correlation >= threshold.
    impute_strategy::Function: Function to impute missing values (default: mean).

Returns:
    DataFrame: The filtered DataFrame.

Example:
    filtered_data = filterColumnsByCorrelation(loader, "target_column", 0.8, true)
"""



# Filtrar columnas por correlación
function filterColumnsByCorrelation(loader::EDALoader, target::String, threshold::Float64, relation::Bool)
    update_cache!(loader)

    # Verificar si el target está en el DataFrame
    if target ∉ names(loader.data)
        throw(ArgumentError("La columna $target no existe en el DataFrame."))
    end

    # Seleccionar columnas numéricas, excluyendo el target
    numeric_cols = filter(col -> eltype(loader.data[!, col]) <: Union{Number, Missing}, names(loader.data))
    numeric_cols = setdiff(numeric_cols, [target])  # Evitar duplicados del target
    numeric_df = select(loader.data, numeric_cols)

    # Validar y limpiar el target
    target_data = skipmissing(loader.data[!, target])
    if isempty(target_data)
        throw(ArgumentError("La columna $target no tiene suficientes datos no faltantes para calcular correlaciones."))
    end
    target_data = collect(target_data)

    # Inicializar lista de columnas a eliminar
    cols_to_drop = String[]

    # Iterar sobre columnas y calcular correlaciones
    for col in names(numeric_df)
        col_data = skipmissing(numeric_df[!, col])

        # Verificar si la columna tiene datos suficientes
        if isempty(col_data)
            continue  # Saltar columnas vacías
        end
        col_data = collect(col_data)

        # Verificar que las longitudes coincidan antes de calcular correlación
        if length(target_data) == length(col_data)
            corr_value = cor(target_data, col_data)

            # Aplicar filtro basado en el parámetro `relation`
            if (relation == false && abs(corr_value) <= abs(threshold)) || (relation == true && corr_value >= threshold)
                push!(cols_to_drop, col)
            end
        end
    end

    # Asegurar que el target esté en el resultado y no se duplique
    cols_to_keep = setdiff(names(loader.data), cols_to_drop)

    # Actualizar el DataFrame con las columnas seleccionadas
    loader.data = select(loader.data, cols_to_keep)
    update_cache!(loader)
    return loader.data
end










"""
Generates a graph representing the correlation network for numeric columns.
Nodes represent columns, and edges represent strong correlations above a threshold.

Args:
    loader::EDALoader: The data loader object.
    threshold::Float64: Minimum correlation value to connect nodes (default: 0.5).

Returns:
    SimpleGraph: The correlation graph.

Example:
    graph = correlation_network(loader, threshold=0.7)
"""



# Función correlation_network
function correlation_network(loader::EDALoader; threshold::Float64 = 0.5, min_degree::Int = 2)
    update_cache!(loader)

    numeric_cols = findall(col -> all(v -> v isa Union{Missing, Number}, col), eachcol(loader.data))
    numeric_data = select(loader.data, numeric_cols)

    if isempty(numeric_data)
        throw(ArgumentError("No hay columnas numéricas válidas para calcular la red de correlaciones."))
    end

    corr_matrix = pairwise(cor, eachcol(numeric_data), skipmissing=:pairwise, symmetric=true)
    column_names = names(numeric_data)

    g = SimpleGraph(length(column_names))
    edge_weights = Dict{Edge{Int}, Float64}()

    for i in 1:size(corr_matrix, 1)
        for j in i+1:size(corr_matrix, 2)
            corr_value = corr_matrix[i, j]
            if abs(corr_value) >= threshold
                add_edge!(g, i, j)
                edge_weights[Edge(i, j)] = corr_value
            end
        end
    end

    degrees = degree(g)
    valid_nodes = findall(d -> d >= min_degree, degrees)
    if isempty(valid_nodes)
        println("No hay nodos con el grado mínimo especificado ($min_degree).")
        return g
    end

    node_map = Dict(valid_nodes .=> 1:length(valid_nodes))
    filtered_graph = SimpleGraph(length(valid_nodes))
    filtered_names = column_names[valid_nodes]

    for edge in edges(g)
        src, dst = Tuple(edge)
        if src in valid_nodes && dst in valid_nodes
            add_edge!(filtered_graph, node_map[src], node_map[dst])
        end
    end

    x, y = spring_layout(filtered_graph)
    labels = [filtered_names[i] for i in 1:length(filtered_names)]
    node_colors = [RGBA(0.1, 0.2, 0.5, 0.8) for _ in 1:length(filtered_names)]
    node_sizes = [5 + degrees[valid_nodes[i]] * 2 for i in 1:length(valid_nodes)]
    lw_factor = 3

    p = plot(
        legend=false,
        size=(1200, 1200),
        xlabel="Coordenada X",
        ylabel="Coordenada Y",
        title="Correlation Network: $(nv(filtered_graph)) Nodos, $(ne(filtered_graph)) Aristas",
        grid=false,
        background_color=:lightgrey
    )

    hline!([0], color=:black, lw=0.5, linestyle=:dash, label="")
    vline!([0], color=:black, lw=0.5, linestyle=:dash, label="")

    for edge in edges(filtered_graph)
        src, dst = Tuple(edge)
        weight = edge_weights[Edge(valid_nodes[src], valid_nodes[dst])]
        color = weight > 0 ? :green : :red
        plot!(
            [x[src], x[dst]], [y[src], y[dst]],
            color=color,
            lw=lw_factor * abs(weight),
            label=""
        )
    end

    scatter!(
        x, y,
        c=node_colors,
        ms=node_sizes,
        label="",
        markerstrokec=:black,
        markerstrokew=0.5
    )

    for i in 1:length(x)
        annotate!(x[i], y[i], text(labels[i], 12, :black))
    end

    annotate!(maximum(x) - 0.2, minimum(y) + 0.3, text("Verde: correlación positiva", 12, :green))
    annotate!(maximum(x) - 0.2, minimum(y) + 0.2, text("Rojo: correlación negativa", 12, :red))
    annotate!(maximum(x) - 0.2, minimum(y) + 0.1, text("Tamaño: grado del nodo", 12, :black))

    display(p)
    return filtered_graph
end












"""
Saves the current state of the DataFrame to a CSV file.

Args:
    loader::EDALoader: The data loader object.
    filepath::String: Path to save the CSV file.

Returns:
    Nothing: Saves the file and prints a success message.

Example:
    save_to_csv(loader, "path/to/output.csv")
"""


function save_to_csv(loader::EDALoader, filepath::String)
    update_cache!(loader)  
    try
        CSV.write(filepath, loader.data)
        println("Archivo guardado exitosamente en: $filepath")
    catch e
        println("Error al guardar el archivo: $e")
    end
end


"""
Performs a linear regression analysis and visualizes the results with a
scatter plot comparing actual vs. predicted values.

Args:
    loader::EDALoader: The data loader object.
    target::Symbol: The target variable.
    predictors::Vector{Symbol}: Predictor variables.

Returns:
    StatisticalModel: The fitted linear regression model.

Example:
    model = linearregression(loader, :target_variable, [:predictor1, :predictor2])
"""


function linearregression(loader::EDALoader, target::String, predictors::Vector{String})
    # Verificar que las columnas especificadas existan
    all_columns = names(loader.data)
    missing_cols = [col for col in [target; predictors] if col ∉ all_columns]
    if !isempty(missing_cols)
        throw(ArgumentError("Las siguientes columnas no están presentes en el DataFrame: $missing_cols"))
    end

    # Eliminar filas con valores faltantes en las columnas de interés
    columns_to_check = [target; predictors]
    data = dropmissing(loader.data, columns_to_check)

    # Crear la fórmula para el modelo de regresión
    target_symbol = Symbol(target)
    predictors_symbols = Symbol.(predictors)
    formula_str = string(target_symbol, " ~ ", join(predictors_symbols, " + "))
    formula = eval(Meta.parse("@formula($formula_str)"))

    # Ajustar el modelo de regresión lineal
    modelo = lm(formula, data)

    # Imprimir los coeficientes del modelo
    println(coeftable(modelo))

    # Agregar los valores predichos al DataFrame
    data[!, :Predicted] = predict(modelo, data)

    # Generar el gráfico de valores reales vs. predichos
    p = scatter(
        data[!, target_symbol],
        data[!, :Predicted],
        xlabel = "Valores reales ($target)",
        ylabel = "Valores predichos ($target)",
        title = "Regresión lineal: Valores reales vs. predichos",
        legend = false
    )
    plot!(x -> x, label="Línea ideal")

    # Mostrar la gráfica explícitamente
    display(p)

    return modelo  # Retorna el modelo ajustado
end




end
