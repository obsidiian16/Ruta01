
# Funciones de uso privado. Para analisis exploratorio de datos y algo de preprocesamiento 
# de datos

from typing import Dict, Optional, Callable, Tuple, Union, List
from numpy import exp
import numpy
from numpy.core.fromnumeric import repeat, shape
import pandas
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as graph_objects
import seaborn as sns
import matplotlib.pyplot as plt

'''
Varios métodos sencillos para crear tramas
'''
from typing import Dict, Optional, Callable, Tuple, Union, List
from numpy import exp
import numpy
from numpy.core.fromnumeric import repeat, shape
import pandas
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as graph_objects

# Establecer el tema predeterminado
template =  graph_objects.layout.Template()
template.layout = graph_objects.Layout(
                                    title_x=0.5,
                                    # border width and size
                                    margin=dict(l=2, r=2, b=2, t=30),
                                    height=400,
                                    # Interaction
                                    hovermode="closest",
                                    # axes
                                    xaxis_showline=True,
                                    xaxis_linewidth=2,
                                    yaxis_showline=True,
                                    yaxis_linewidth=2,
                                    # Pick a slightly different P.O.V from default
                                    # this avoids the extremities of the y and x axes
                                    # being cropped off
                                    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.1))
                                    )

template.data.scatter = [graph_objects.Scatter(marker=dict(opacity=0.8))]
template.data.scatter3d = [graph_objects.Scatter3d(marker=dict(opacity=0.8))]
template.data.surface = [graph_objects.Surface()]
template.data.histogram = [graph_objects.Histogram(marker=dict(line=dict(width=1)))]
template.data.box = [graph_objects.Box(boxpoints='outliers', notched=False)]


pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template"

# Colores de tendencia
# Tenga en cuenta que el texto de este curso a menudo se refiere explícitamente a los colores
# como "mirar la línea roja". Cambiar la variable a continuación puede 
# resultar en que este texto sea inconsistente
colours_trendline = px.colors.qualitative.Set1  

# Funciones de agregacion para graficas 

def _to_human_readable(text:str):
    '''
    Converts a label into a human readable form
    '''
    return text.replace("_", " ")

def _prepare_labels(df:pandas.DataFrame, labels:List[Optional[str]], replace_nones:bool=True):
    '''
    Ensures labels are human readable.
    Automatically picks data if labels not provided explicitly
    '''

    human_readable = {}

    if isinstance(replace_nones, bool):
        replace_nones = [replace_nones] * len(labels)

    for i in range(len(labels)):
        lab = labels[i]
        if replace_nones[i] and (lab is None):
            lab = df.columns[i]
            labels[i] = lab

        # make human-readable
        if lab is not None:
            human_readable[lab] = _to_human_readable(lab)

    return labels, human_readable

# ---------------- GRAFICAS DE DATOS ----------------------

# ----------------------- DATOS CUANTITATIVOS -----------------------------------

def distribution_Data(label):
    
    '''
    Muetra dos visualizaciones: - Distribucion de datos (Histograma)
                                - Diagrama de Caja

    label[pd.Series o pd.Dataframe] : Datos que se usaran, debe ser un variable cuantitativa con tipo
                                      de datos float o int. Si:
                                      - label[pd.Dataframe] = Debe contener solo una columna con la variable a usar
                                      - laber[pd.Series]    = No hay inconveniente                                
                                
    NOTA: 
    En grafica de histograma (datos adicionales):
            - Datos minimos y maximos (color: Magenta)
            - Media (color: cyan o celeste)
            - Mediana (color: verde)

    
    '''

    import pandas as pd
    import matplotlib.pyplot as plt

    # %matplotlib inline

    if (type(label) == type(pd.DataFrame()) | type(label) == type(pd.Series())):
        pass
    else:
        label = pd.Series(label)

    
    # creamos la figura 
    fig, ax = plt.subplots(2, 1, figsize = (9,12))  # Generamos la figura

    # SUBPLOT 1: Ploteo de histogramas (distribucion de datos)
    ax[0].hist(label, bins = 100)
    ax[0].set_ylabel('Frecuencia')

    # Lineas de referencia para los datos
    ax[0].axvline(label.min(), color = 'magenta', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(label.max(), color = 'magenta', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(label.mean(), color = 'cyan', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(label.median(), color = 'green', linestyle = 'dashed', linewidth = 2)

    # SUBPLOT 2: Ploteo de diagrama de caja
    ax[1].boxplot(label, vert= False)   # Diagrama de caja horizontal
    ax[1].set_xlabel(str(label.name))

    # Añadimos el titulo a la figura
    fig.suptitle(str(label.name)+ ' Distribution')
    fig.show()

    return fig

# ----------------------- DATOS CUALITATIVOS -----------------------------------

# ESTE TIPO DE GRAFICAS AYUDA A VER LA RELACION ENTRE VARIALES CUANTITATIVAS
# Y VARIABLES CUALITATIVAS

def box_and_whisker(df:pandas.DataFrame,
                label_x:Optional[str]=None,
                label_y:Optional[str]=None,
                label_x2:Optional[str]=None,
                title=None,
                show:bool=False):
    '''
    Crea un diagrama de caja y bigotes y, opcionalmente, lo muestra. Devuelve la cifra de esa parcela.

     Tenga en cuenta que si llama a esto desde cuadernos jupyter y no captura la salida
     aparecerá en la pantalla como si se hubiera llamado `.show()`

    df[pd.Dataframe]    : Datos formato pandas
    label_x[list]       : Por qué agrupar. Predeterminado a Ninguno
    label_y[list]       : Qué trazar en el eje y. Predeterminado para contar df.columns[0]
    label_x2            : Si se proporciona, divide los diagramas de caja en más de 2 por valor x, cada uno con su propio color
    title               : Título de la trama
    show                : Aparece en pantalla. NB que esto no es necesario si se llama desde un
                          portátil y la salida no se captura

    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_x2], replace_nones=[False, True, False])

    fig = px.box(df,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    color=label_x2,
                    labels=axis_labels,
                    title=title)

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig

def histogram(df:pandas.DataFrame,
                label_x:Optional[str]=None,
                label_y:Optional[str]=None,
                label_colour:Optional[str]=None,
                nbins:Optional[int]=None,
                title=None,
                include_boxplot=False,
                histfunc:Optional[str]=None,
                show:bool=False):
    '''
    Crea un histograma 2D y, opcionalmente, lo muestra. Devuelve la cifra de ese histograma.

    Tenga en cuenta que si llama a esto desde cuadernos jupyter y no captura la salida
    aparecerá en la pantalla como si se hubiera llamado `.show()`

    df          : Los datos
    label_x     : Por qué agrupar. El valor predeterminado es df.columns[0]
    label_y     : si se proporciona, la suma de estos números se convierte en el eje y. Predeterminado para contar de label_x
    label_colour: si se proporciona, crea un histograma apilado, dividiendo cada barra por esta columna
    title       : título de la trama
    nbins       : el número de contenedores a mostrar. Ninguno para automático
    histfunc    : Cómo calcular y. Ver plotly para opciones
    show        : aparece en pantalla. NB que esto no es necesario si se llama desde un
            portátil y la salida no se captura
    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_colour], replace_nones=[True, False, False])


    fig = px.histogram(df,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        nbins=nbins,
                        color=label_colour,
                        labels=axis_labels,
                        title=title,
                        marginal="box" if include_boxplot else None,
                        histfunc=histfunc
                        )

    # Set the boxplot notches to False by default to deal with plotting bug
    # But only call this line if the user wants to include a boxplot
    if include_boxplot:
        fig.data[1].notched = False

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig

def multiple_histogram(df:pandas.DataFrame,
                label_x:str,
                label_group:str,
                label_y:Optional[str]=None,
                histfunc:str='count',
                nbins:Optional[int]=None,
                title=None,
                show:bool=False):
    '''
    Crea un histograma 2D y, opcionalmente, lo muestra. Devuelve la cifra de ese histograma.

    Tenga en cuenta que si llama a esto desde cuadernos jupyter y no captura la salida
    aparecerá en la pantalla como si se hubiera llamado `.show()`

    df          : Los datos
    label_x     : Por qué agrupar. El valor predeterminado es df.columns[0]
    label_y     : si se proporciona, la suma de estos números se convierte en el eje y. Predeterminado para contar de label_x
    title      : título de la trama
    nbins       : el número de contenedores a mostrar. Ninguno para automático
    show     : aparece en pantalla. NB que esto no es necesario si se llama desde un
            portátil y la salida no se captura

    '''

    assert (histfunc != 'count') or (label_y == None), "Set histfunc to a value such as sum or avg if using label_y"

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df,  [label_x, label_y, label_group], replace_nones=[True, False, False])

    fig = graph_objects.Figure(layout=dict(
                                    title=title,
                                    xaxis_title_text=axis_labels[label_x],
                                    yaxis_title_text=histfunc if label_y is None else (histfunc + " of " + axis_labels[label_y]))
                                )

    group_values = sorted(set(df[label_group]))

    for group_value in group_values:
        dat = df[df[label_group] == group_value]
        x = dat[selected_columns[0]]

        if label_y is None:
            y = None
        else:
            y = dat[selected_columns[1]]

        fig.add_trace(graph_objects.Histogram(
            x=x,
            y=y,
            histfunc=histfunc,
            name=group_value, # name used in legend and hover labels
            nbinsx=nbins))

    #Place legend title
    fig.update_layout(legend_title_text=label_group)

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig

# --------------------- EXPLORACION DE DATOS CUANTITATIVOS ---------------------------------

def set_cuantiles(data: pandas.DataFrame,cuantils_list:List[Optional[float]] = None):
    
    '''Muestra los cuantiles de un conjunto de datos
    
    data[Dataframe]     : Set de datos tipo pandas (datos cualitativos)
    cuantils_list[list] : Cuantiles que desean visualizarse (solo valors flotantes entre 0 y 1)

    Default:
        cuantils_list = None (no hay ingreso de lista)
            ---> por defecto:  cuantiles: [0.0, 0.1, 0.25, 0.50, 0.75, 0.95, 1]

    '''
    
    import pandas as pd

    # Validadores
    if type(data) != type(pd.DataFrame()):
        return 'Ingrese set de datos tipo pandas'
    # ------------------------------------------------------------------

    if cuantils_list is None:
        percentiles = [0.0, 0.1, 0.25, 0.50, 0.75, 0.95, 1]
    else:
        percentiles = []
        for quantil in cuantils_list:
            if type(quantil) == float:
                if (quantil >= 0) | (quantil <= 1):
                    percentiles.append(quantil)
            else:
                return 'Los cuantiles deben ser numeros flotantes entre 0 y 1'

    df_data_perc = pd.DataFrame(data.quantile(percentiles))
    return df_data_perc

def mapa_correlaciones(data:pandas.DataFrame, corr_type:Optional[str]=None):
    
    '''Muestra la correlacion entre variables generando un mapa de calor y
    ademas muestra el valor de correlacion entre variables
    
    data[Dataframe]     : Set de datos tipo pandas
    corr_type[String]   : tipo de correlacion que usaremos (tenemos 3 opciones)
                                - pearson
                                - kendall
                                - spearman
    
    Default:
        Correlacion de Pearson
    '''

     # Validadores
    if type(data) != type(pandas.DataFrame()):
        return 'Ingrese set de datos tipo pandas'
    # ------------------------------------------------------------------

    corr_opcions = ['pearson', 'kendall', 'spearman']
    if corr_type is None:
        corr = data.corr(method= 'pearson')
    else: 
        if corr_type in corr_opcions:
            corr = data.corr(method = corr_type)
        else:
            return 'El tipo de correlacion ingresada es incorrecta'
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
           xticklabels= data.columns, 
           yticklabels= data.columns,
           cmap= 'coolwarm').set_title('Mapa de Correlaciones')
    plt.show()


# ------------------------------ RECORTE DE DATOS --------------------------------------------

def cut_cuantiles(data:pandas.DataFrame, name_column: str, rango: float, type_cut:bool = True , less_equal:bool = True):
    
    """Selecciona datos menores o mayores para cierto umbral de cuantiles
    data [Dataframe]   : Set de datos en formato pandas
    name_column [str]   : Nombre de la columna a recortar (debe encontrarse dentro del set de datos)
    type_cut [Bool]     : Tipo de recorte:
                            True --> rangos cuantilicos
                            False --> rangos numericos 
    rangos [float]      : Rango de recorte 
    less_equal [bool]   : Recorte:
                            True  --> menor igual que el rango
                            False --> mayor igual que el rango

    Por defecto:
        type_cute --> True, less_equal --> True

    """
    
    columns_names = data.columns

    # Conformidad con set de datos en formato DataFrame de pandas
    if type(data) == type(pandas.DataFrame()):
        if name_column in columns_names:
            if type_cut == True:    # Recorte por cuantil
                
                if (rango >= 0.0) or (rango <= 1.0): # Rangos de cuantil en decimal [0,1]
                    if less_equal == True:    # Menor que <=
                        cuantil_range = data[name_column].quantile(rango)
                        return data[data[name_column] <= cuantil_range]
                    
                    else:       #less_equal == False (>=)
                        cuantil_range = data[name_column].quantile(rango)
                        return data[data[name_column] >= cuantil_range]

                else: 
                    return 'Rango incorrecto, ingrese numeros de 0 a 1 (flotantes)'

            else:   # type_cut == False (Recorte por rango)
                if (rango >= data[name_column].min()) or (rango <= data[name_column].max()):
                    if less_equal == True:    # Menor que <=
                        return data[data[name_column] <= rango]
                    
                    else:       #less_equal == False (>=)
                        return data[data[name_column] >= rango]

        else:
            return 'Nombre de columna no encontrado'

    else: 
        return 'ingresar instancia de datos tipo Dataframe de Pandas'





# CREAR FUNCION PARA RECORTAR OUTLIERS DE SET DE DATOS (por desviacion estandar)
# PASTILLERO INTELIGENTE XD (CON ALARMA INCLUIDA)


    