# Informe técnico · RIW Neural LAB v0.1

## Propósito
Implementar y comprobar una MLP mínima que aprenda XOR sin bibliotecas de redes neuronales.

## Arquitectura
Dos entradas binarias, cuatro unidades ocultas con activación sigmoide y una salida sigmoide. Total: 17 parámetros ajustables. Inicialización pseudoaleatoria con semilla 42.

## Entrenamiento
20 000 épocas, tasa de aprendizaje 1.0, cuatro patrones por lote, pérdida cuadrática media. Se calculan gradientes por regla de la cadena (retropropagación) y se actualizan pesos y sesgos.

## Evidencias
Los valores observados de consola se documentan en `resultados-v0.1.md`. La gráfica muestra exclusivamente cinco puntos registrados; no implica que se hayan guardado los valores de todas las épocas.

## Alcance y limitaciones
XOR es un problema cerrado de cuatro entradas posibles. Acertar las cuatro no demuestra capacidad de generalizar a problemas nuevos ni equivale a un detector de intrusiones. No hay validación externa ni comparación con métodos tradicionales en esta versión.

## Próxima versión
Registrar automáticamente métricas por época, añadir pruebas unitarias de gradientes y comparar con una solución lógica convencional. Futuras pruebas de seguridad solo con datos sintéticos y métricas de falsos positivos.
