//  VARIABLES DE VISUALIZACIÓN
var ndvi_palette = 'FFFFFF, CE7E45, DF923D, F1B555, FCD163, 99B718, 74A901, 66A000, 529400, ' + '3E8601, 207401, 056201, 004C00, 023B01, 012E01, 011D01, 011301';

//  CONFIGURACIÓN DE LA SERIE DE TIEMPO Y ZONAS
var anios = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018];

var zonas = [
  {nombre: 'ZONA_1', geometria: geometry},
  {nombre: 'ZONA_2', geometria: geometry2},
  {nombre: 'ZONA_3', geometria: geometry3},
  {nombre: 'ZONA_COMPLETA', geometria: geometry4}
];

// Lista de 11 meses exactos para etiquetar las columnas (Marzo a Enero)
var offsets = ee.List.sequence(0, 10);
var nombresMeses = ee.List(['Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic', 'Ene']);

//  EL MOTOR DE EXTRACCIÓN MODIS (250m)
for (var i = 0; i < anios.length; i++) {
  var anio = anios[i];
  var fechaInicioBase = ee.Date.fromYMD(anio - 1, 3, 1); // Comienza en Marzo del año anterior

  var crearBandaMensual = function(offset) {
    var n = ee.Number(offset);
    var mesInicio = fechaInicioBase.advance(n, 'month');
    var mesFin = mesInicio.advance(1, 'month');

    // Colección MODIS Terra Vegetation Indices (Cada 16 días, resolución 250m)
    var coleccionTotal = ee.ImageCollection("MODIS/061/MOD13Q1")
                         .filterDate(mesInicio, mesFin)
                         .select('NDVI');

    // Resguardo matemático (vacío con formato float)
    var dummy = ee.Image.constant(0).toFloat().rename('NDVI').updateMask(0);
    
    // Filtro de calidad: Tomamos el valor máximo del mes (el día más despejado y con más verdor)
    var mesFinal = ee.Image(ee.Algorithms.If(
      coleccionTotal.size().eq(0),
      dummy,
      coleccionTotal.max()
    ));

    return mesFinal;
  };

  // Mapeamos los 11 meses y los convertimos en un mapa multibanda (un archivo con 11 capas adentro)
  var coleccionMensual = ee.ImageCollection.fromImages(offsets.map(crearBandaMensual));
  var imagenMultibanda = coleccionMensual.toBands();

  // Renombramos las bandas internamente para que QGIS sepa cuál es cuál
  var nombresFinales = offsets.map(function(n) {
      return ee.String('NDVI_').cat(ee.String(nombresMeses.get(n)));
  });
  imagenMultibanda = imagenMultibanda.rename(nombresFinales);

// EXPORTACIÓN MASIVA A GOOGLE DRIVE
  for (var j = 0; j < zonas.length; j++) {
    var zona = zonas[j];
    var nombreArchivo = 'NDVI_MODIS_' + anio + '_' + zona.nombre;

    Export.image.toDrive({
      image: imagenMultibanda.clip(zona.geometria),
      description: nombreArchivo,
      fileNamePrefix: nombreArchivo,
      folder: 'GEE_MODIS_Tesis',
      scale: 250, // Resolución nativa de MODIS
      maxPixels: 1e13,
      region: zona.geometria
    });
  }
}