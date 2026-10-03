# Animación del Forward Propagation (TP Sistemas de IA · Wine Quality)

Celdas para agregar a la notebook de Google Colab **sin modificar las existentes**:

| Archivo | Tipo de celda |
|---|---|
| `celda_markdown_animacion.md` | Celda de **texto** (Markdown) |
| `celda_codigo_animacion_forward.py` | Celda de **código** |
| `preview_animacion.gif` | Vista previa generada con datos sintéticos (solo de referencia) |

## Dónde colocarlas

Después del entrenamiento (`history_2 = modelo_2.fit(...)`) y **antes** de la evaluación final.
Primero va la celda Markdown y debajo la de código.

## Requisitos

- `modelo_2` debe estar **entrenado**: la animación usa sus pesos aprendidos. Con un modelo sin entrenar
  se ejecuta igual, pero muestra pesos aleatorios.
- Variables usadas (solo lectura): `modelo_2`, `X_test`, `y_test` y, si existe, `X` (para los nombres de columnas).
- No reentrena, no modifica modelos ni datos y no guarda archivos.
- Genera unos 8 MB de HTML (70 cuadros JPEG, 14 s por ciclo) con controles de reproducción.

## Opciones

```python
animar_forward_propagation(modelo_2, X_test, y_test, indice=15)          # ejemplo fijo
animar_forward_propagation(modelo_2, X_test, y_test, clase_preferida=0)  # un vino NO BUENO bien clasificado
animar_forward_propagation(modelo_1, X_test, y_test, nombre_modelo="modelo_1")
```
