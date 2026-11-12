# Guía didáctica

Una neurona artificial multiplica entradas por pesos, suma un sesgo y aplica una función de activación. La capa oculta permite aprender relaciones no lineales como XOR.

| A | B | XOR |
|---|---|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Experimento:** ejecutar `neural_xor.py`, observar el error y cambiar la semilla aleatoria o la tasa de aprendizaje. No todas las configuraciones garantizan convergencia. Comparar resultados sin modificar el conjunto de datos.

**Pregunta para aprender:** ¿por qué una sola neurona sigmoide sin capa oculta no puede separar los cuatro puntos XOR mediante una frontera lineal?
