# Análisis exploratorio de datos vitivinícolas 
## Introduccion
Este informe explicara el análisis y procesamiento de datos realizado a través de un script en python siendo este segmentado en diferentes etapas. Dicho trabajo estará dividido por las tareas de carga de archivo, el análisis exploratorio, el tratamiento de datos faltantes y el tratamiento de datos atípicos. 
Cada segmento describirá detalladamente los pasos realizados en el código sirviendo como un manual, sumado a que se justificara todas las decisiones tomadas y se comentara alternativas que se podrían haber realizado como así también de las descartadas.


## Carga de archivo

Utilizamos Python para cargar el archivo winemag-data-130k-v2.csv.
Para comenzar a realizar el análisis vamos a cargar el archivo csv el cual es nuestra principal y única fuente de datos. Sin embargo, en la práctica este archivo se encuentra en formato xlsx o sea el formato utilizado por Microsoft Excel para almacenar hojas de cálculo.  Al tratarse de un gran conjunto de datos, por la simplicidad de su formato y por la cantidad de librerías disponibles en python para esta forma , vamos a utilizar el formato csv para la carga de datos. Para ello tenemos dos opciones para transformar el archivo fuente a csv, por un lado podemos realizarlo de manera manual desde Microsoft Excel  y por otro a través de otras líneas de código. Al tratarse de una acción que se realizara por única vez, con un formato simple y para no agregar mas líneas al script se optara por la opción manual y se le asignara el mismo nombre de archivo. Si hubira existido datos mas complejos en el archivo Excel o si la tarea tendría que repetirse se podría haber utilizado la función de pandas “pd.read_excel” y “df.to_csv”.
Aclarado esto vamos a realizar la carga del archivo a través de  la librería pandas y la función :

 df = pd.read_csv('winemag-data-130k-v2.csv') 
 
  ### Revisar el formato del archivo, asegurándose de que se haya cargado correctamente.
Como bien aclaramos en el punto anterior realizamos la carga del archivo en formato cvs. La carga del archivo se realiza a través de una función llamada cargar_datos la cual es llamada como las demás en la función principal main ().
Esta función carga el archivo a través de pandas con su pd.read_csv como mencionamos anteriormente sumado a que se tuvo en cuenta la separación por punto y coma (en caso de que el archivo csv muestre esta separación, por defecto es con la coma). Además se tuvo en cuenta la codificación UTF-8 (Unicode Transformation Format - 8 bits) la cual  es una de las codificaciones de texto más utilizadas en el mundo. Por ultimo también no se tomó como índice a la primera columna del dataset (Unnamed: 0 ) ya que nos servira para corroborar algunos datos mal cargados mas adelante.
Esta función posee un try al principio para corroborar si el archivo se cargó correctamente con una advertencia y la descripción del error en caso de que el archivo haya tenido algún inconveniente(o sea si entra en el except)

### Verificar la cantidad de datos, número de columnas y tipos de datos presentes en el conjunto de datos.

Para realizar esta tarea se utilizó la función inspeccionYcorreccion . En ella primero mostramos los datos que tenemos en nuestro dataset a través del comando shape de pyhton y la estructuramos en columnas columnas para que sean visibles de manera adecuada en streamlit. Podemos notar que contamos con 129975 filas y 14 columnas (recordemos que NO tomamos la primera como índice).
Para verificar los tipos de datos utilizamos el comando info de pandas. Este método imprime información sobre un “DataFrame”, incluido el índice, el tipo de dato y las columnas, los valores no nulos y el uso de memoria. 

