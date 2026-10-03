## Visualización del Forward Propagation en `modelo_2`

En las primeras clases hicimos a mano el cálculo de una neurona: cada neurona recibe entradas, las multiplica por sus **pesos**, suma un **sesgo** y aplica una **función de activación**:

$$z = \sum_{i} x_i \, w_i + b \qquad\qquad a = f(z)$$

Una red densa repite exactamente ese cálculo para muchas neuronas y capas. Keras lo hace automáticamente. La animación siguiente muestra ese proceso, llamado **Forward Propagation**, con **un vino real de `X_test`** pasando por el `modelo_2` ya entrenado:

| Paso | Capa | Cálculo |
|---|---|---|
| 1 | **Entrada** (12 características normalizadas) | no se calcula nada; los valores pasan a la capa siguiente |
| 2 | **Capa oculta 1**: 12 → 64 neuronas | $z = \sum x_i w_i + b$, $\;a = \mathrm{ReLU}(z) = \max(0, z)$ |
| 3 | **Capa oculta 2**: 64 → 32 neuronas | igual, ahora las entradas son las activaciones anteriores |
| 4 | **Capa oculta 3**: 32 → 16 neuronas | igual |
| 5 | **Salida**: 16 → 1 neurona | $p = \sigma(z) = \dfrac{1}{1 + e^{-z}}$, una probabilidad entre 0 y 1 |
| 6 | **Resultado** | si $p \geq 0.5$: **VINO BUENO**; si no, **VINO NO BUENO** |

**Cómo leer la animación**
- Todos los valores son **reales**: activaciones de cada capa del modelo entrenado, pesos aprendidos y probabilidad final calculada con `modelo_2.predict()`.
- **Color y tamaño** de cada neurona: magnitud de su activación. Las neuronas con **a = 0** quedaron "apagadas" por ReLU.
- **Líneas**: aporte $a_i \cdot w_i$ que llega a cada neurona (azul = positivo, rojo = negativo; más gruesa = mayor aporte).
- La **neurona marcada en naranja** muestra su cálculo numérico completo en el recuadro inferior, igual que lo hacíamos a mano.
- *Representación simplificada:* de cada capa oculta se dibujan solo 10 neuronas (las 7 más activas y las 3 menos activas). Los cálculos usan **todas** las neuronas: 64 + 32 + 16 + 1 = 113.
