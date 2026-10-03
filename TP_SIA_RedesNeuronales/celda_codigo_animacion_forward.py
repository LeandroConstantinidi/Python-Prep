# =====================================================================
# ANIMACIÓN DIDÁCTICA · FORWARD PROPAGATION DE UN VINO REAL EN modelo_2
# ---------------------------------------------------------------------
# - Ejecutar DESPUÉS de entrenar modelo_2 (usa sus pesos ya aprendidos).
# - No reentrena, no modifica modelos ni datos y no guarda archivos.
# - Todo queda dentro de una función para no pisar variables de la notebook.
# =====================================================================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from IPython.display import HTML, display


def animar_forward_propagation(modelo, X_prueba, y_prueba,
                               indice=None,          # None = selección automática
                               clase_preferida=1,    # 1: buscar un vino BUENO bien clasificado
                               umbral=0.5,
                               max_visibles=10,      # neuronas dibujadas por capa oculta
                               frames_por_paso=10,
                               intervalo_ms=200,     # (5 pasos x 10 + resultado x 20) x 200 ms = 14 s
                               nombre_modelo="modelo_2"):
    # ---------- 1) Datos como numpy (copias: X_test / y_test no se tocan) ----------
    Xt = np.array(X_prueba, dtype="float32")
    if Xt.ndim == 1:
        Xt = Xt.reshape(1, -1)
    yt = np.array(y_prueba).astype(int).ravel()       # posicional: evita índices de pandas
    n_feat = Xt.shape[1]

    nombres = ["acidez_fija", "acidez_volatil", "acido_citrico", "azucar_residual",
               "cloruros", "dioxido_azufre_libre", "dioxido_azufre_total", "densidad",
               "ph", "sulfatos", "alcohol", "tipo_vino"]
    X_global = globals().get("X")                     # X original (DataFrame) si existe
    for fuente in (X_prueba, X_global):
        if hasattr(fuente, "columns") and len(fuente.columns) == n_feat:
            nombres = [str(c) for c in fuente.columns]
            break
    if len(nombres) != n_feat:
        nombres = [f"x{i + 1}" for i in range(n_feat)]

    # ---------- 2) Verificar que el modelo esté construido ----------
    if not modelo.built:
        modelo(Xt[:1])        # solo construye el grafo; no entrena ni cambia pesos aprendidos

    # ---------- 3) Elegir el ejemplo ----------
    # Criterio: entre los ejemplos BIEN clasificados de la clase preferida, el de
    # probabilidad mediana (ni dudoso ni extremo). Si no hay, cualquier bien
    # clasificado; si tampoco hay, el primer ejemplo de X_test.
    probs = modelo.predict(Xt, verbose=0).ravel()
    pred = (probs >= umbral).astype(int)
    if indice is None:
        ok = np.where((pred == yt) & (yt == clase_preferida))[0]
        if len(ok) == 0:
            ok = np.where(pred == yt)[0]
        if len(ok) > 0:
            orden = ok[np.argsort(probs[ok])]
            indice = int(orden[len(orden) // 2])
            criterio = "bien clasificado, probabilidad mediana de su clase"
        else:
            indice = 0
            criterio = "primer ejemplo (no se halló uno bien clasificado)"
    else:
        criterio = "elegido manualmente"
    x_ej = Xt[indice:indice + 1]                      # shape (1, 12)

    # ---------- 4) Activaciones reales de cada capa ----------
    try:   # Opción A: modelo intermedio de Keras con todas las salidas
        if not modelo.inputs:
            raise ValueError("modelo sin input simbólico")
        entrada = modelo.inputs[0] if len(modelo.inputs) == 1 else modelo.inputs
        extractor = tf.keras.Model(inputs=entrada,
                                   outputs=[c.output for c in modelo.layers])
        salidas = extractor(x_ej, training=False)
        salidas = salidas if isinstance(salidas, (list, tuple)) else [salidas]
        salidas = [np.array(s) for s in salidas]
        metodo = "tf.keras.Model intermedio"
    except Exception:   # Opción B (compatible con cualquier Sequential): capa por capa
        salidas, h = [], tf.convert_to_tensor(x_ej)
        for c in modelo.layers:
            h = c(h, training=False)
            salidas.append(np.array(h))
        metodo = "capa por capa (Sequential)"

    entradas_capa = [x_ej] + salidas[:-1]
    capas = [{"n": n_feat, "a": x_ej[0].astype(float), "tipo": "entrada"}]
    for c, ent, sal in zip(modelo.layers, entradas_capa, salidas):
        if not isinstance(c, tf.keras.layers.Dense):
            continue                                  # (p. ej. Dropout): no se dibuja
        W, b = [np.array(p, dtype=float) for p in c.get_weights()]
        a_prev = ent[0].astype(float)
        capas.append({"n": W.shape[1], "W": W, "b": b, "a_prev": a_prev,
                      "z": a_prev @ W + b,            # z = Σ(xᵢ·wᵢ) + b  (cálculo "a mano")
                      "a": sal[0].astype(float),      # salida real de Keras
                      "tipo": c.get_config().get("activation", "")})

    p_predict = float(modelo.predict(x_ej, verbose=0).ravel()[0])  # probabilidad REAL
    clase_pred = int(p_predict >= umbral)
    clase_real = int(yt[indice])
    p_manual = float(capas[-1]["a"][0])

    # ---------- 5) Neuronas visibles (representación simplificada) ----------
    # Capas grandes: se dibujan las (max_visibles-3) más activas + las 3 menos activas,
    # en su orden original. Los cálculos usan TODAS las neuronas.
    for L in capas:
        if L["n"] <= max(max_visibles, 12):
            L["vis"] = np.arange(L["n"])
        else:
            orden = np.argsort(L["a"])
            L["vis"] = np.sort(np.concatenate([orden[:3], orden[-(max_visibles - 3):]]))
        L["ej"] = int(L["vis"][np.argmax(L["a"][L["vis"]])])   # neurona de ejemplo
        L["norm"] = L["a"] / max(np.abs(L["a"]).max(), 1e-8)   # sin divisiones por cero

    # ---------- 6) Figura ----------
    n_pasos = len(capas) + 1                          # entrada + capas Dense + resultado
    XS = np.linspace(0.20, 0.86, len(capas))
    def ys(k):
        if k == 1:
            return np.array([0.5])
        alto = 0.84 * (k - 1) / 11
        return np.linspace(0.5 + alto / 2, 0.5 - alto / 2, k)
    for L, x in zip(capas, XS):
        L["pos"] = np.column_stack([np.full(len(L["vis"]), x), ys(len(L["vis"]))])

    C_POS, C_NEG, C_ANAR = "#2166ac", "#d6604d", "#e08214"
    cm_in, cm_hid, cm_out = plt.cm.BrBG, plt.cm.YlGnBu, plt.cm.RdYlGn
    GRIS_F, GRIS_B = np.array([0.93, 0.93, 0.93, 1]), np.array([0.75, 0.75, 0.75, 1])

    fig = plt.figure(figsize=(13, 8), dpi=80, facecolor="white")
    ax = fig.add_axes([0.0, 0.28, 1.0, 0.62])
    ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1.12); ax.axis("off")
    ax_txt = fig.add_axes([0.015, 0.05, 0.60, 0.215]); ax_txt.axis("off")
    ax_fn = fig.add_axes([0.655, 0.085, 0.125, 0.14])
    ax_res = fig.add_axes([0.815, 0.05, 0.17, 0.21])
    ax_txt.add_patch(Rectangle((0, 0), 1, 1, transform=ax_txt.transAxes,
                               facecolor="#f7f7f7", edgecolor="#d9d9d9"))
    try:
        renderer = fig.canvas.get_renderer()        # para medir textos (backend Agg / inline)
        ancho_txt = ax_txt.get_window_extent(renderer).width * 0.96
    except Exception:
        renderer, ancho_txt = None, None

    def ajustar(txt, tam, ancho):   # achica la fuente si la línea no entra en su espacio
        txt.set_fontsize(tam)
        if renderer is None:
            return
        while txt.get_window_extent(renderer).width > ancho and tam > 6:
            tam -= 0.5
            txt.set_fontsize(tam)

    fig.text(0.5, 0.965, f"Forward Propagation de un vino real a través de {nombre_modelo}",
             ha="center", fontsize=15, weight="bold")
    subtitulo = fig.text(0.5, 0.925, "", ha="center", fontsize=12.5, color="#333333")
    fig.text(0.5, 0.006, f"Representación simplificada: en cada capa oculta se dibujan {max_visibles} neuronas "
             f"(las {max_visibles - 3} más activas y las 3 menos activas), pero los cálculos usan todas.\n"
             "Color y tamaño = activación real de cada neurona.  Líneas = aporte aᵢ·wᵢ que recibe la capa "
             "(azul = positivo, rojo = negativo; más gruesa = mayor aporte).",
             ha="center", va="bottom", fontsize=7.5, color="#666666", linespacing=1.3)

    franja = Rectangle((0, -0.07), 0.13, 1.08, facecolor="#fff3dc", edgecolor="none", zorder=0)
    ax.add_patch(franja)

    titulos = ["Entrada"] + [f"Capa oculta {k}" for k in range(1, len(capas) - 1)] + ["Salida"]
    nombre_fn = {"relu": "ReLU", "sigmoid": "Sigmoide"}
    funciones = ["valores normalizados"] + [nombre_fn.get(L["tipo"], L["tipo"]) for L in capas[1:]]
    cabeceras = []
    for k, (L, x) in enumerate(zip(capas, XS)):
        cabeceras.append(ax.text(x, 1.075, f"{titulos[k]}\n", ha="center", va="center",
                                 fontsize=11.5, weight="bold"))
        ax.text(x, 1.035, funciones[k], ha="center", va="center", fontsize=9, color="#555555")
        nv = len(L["vis"])
        pie = f"{L['n']} neuronas" if L["n"] > 1 else "1 neurona"
        if nv < L["n"]:
            pie = f"{L['n']} neuronas reales\n(se muestran {nv})"
        ax.text(x, -0.075, pie, ha="center", va="center", fontsize=9, color="#555555")

    # Conexiones (solo entre neuronas visibles): base gris + "flujo" coloreado con el aporte real
    base, flujo, aportes = [], [], []
    for k in range(1, len(capas)):
        A, B, L = capas[k - 1]["pos"], capas[k]["pos"], capas[k]
        segs = [(A[i], B[j]) for i in range(len(A)) for j in range(len(B))]
        c = np.array([L["a_prev"][vi] * L["W"][vi, vj] for vi in capas[k - 1]["vis"] for vj in L["vis"]])
        cn = np.abs(c) / max(np.abs(c).max(), 1e-8)
        aportes.append((np.array(segs), cn, np.where(c >= 0, 0, 1)))
        base.append(ax.add_collection(LineCollection(segs, colors="#dddddd", linewidths=0.4, zorder=1)))
        flujo.append(ax.add_collection(LineCollection(segs, linewidths=0.3 + 2.8 * cn, zorder=2)))

    nodos = [ax.scatter(L["pos"][:, 0], L["pos"][:, 1], s=100, zorder=3) for L in capas]
    anillo = ax.scatter([0], [0], s=700, facecolors="none", edgecolors=C_ANAR, linewidths=2.5, zorder=4)

    textos_val = []
    for k, L in enumerate(capas):
        lista = []
        for (x, y), vi in zip(L["pos"], L["vis"]):
            if k == 0:
                ax.text(0.015, y, nombres[vi], ha="left", va="center", fontsize=9.5, color="#333333")
                lista.append(ax.text(x - 0.028, y, "", ha="right", va="center", fontsize=9.5,
                                     family="monospace"))
            elif k == len(capas) - 1:
                lista.append(ax.text(x + 0.03, y, "", ha="left", va="center", fontsize=12, weight="bold"))
            else:
                lista.append(ax.text(x + 0.02, y, "", ha="left", va="center", fontsize=8, color="#333333"))
        textos_val.append(lista)

    linea = [ax_txt.text(0.02, 0.86, "", fontsize=11.5, weight="bold", va="center", transform=ax_txt.transAxes),
             ax_txt.text(0.02, 0.62, "", fontsize=13, va="center", transform=ax_txt.transAxes),
             ax_txt.text(0.02, 0.35, "", fontsize=10, va="center", transform=ax_txt.transAxes),
             ax_txt.text(0.02, 0.13, "", fontsize=9.5, va="center", color="#444444", transform=ax_txt.transAxes)]

    # ---------- 7) Textos de cada paso ----------
    def ejemplo_calculo(k):     # z de la neurona naranja, con sus 2 aportes más grandes
        L = capas[k]
        j = L["ej"]
        top = np.argsort(-np.abs(L["a_prev"] * L["W"][:, j]))[:2]   # los 2 aportes más grandes
        t = " + ".join(f"({L['a_prev'][i]:+.2f})·({L['W'][i, j]:+.2f})" for i in top)
        resto = len(L["a_prev"]) - 2
        return j, f"z = {t} + … ({resto} términos más) + b({L['b'][j]:+.2f}) = {L['z'][j]:.2f}"

    def textos_paso(s):
        if s == 0:
            return (f"Paso 1/{n_pasos} · ENTRADA: ingresan las {n_feat} características del vino",
                    [f"Ejemplo real: fila {indice} de X_test ({criterio}).",
                     r"$\mathbf{x} = (x_1,\ x_2,\ \ldots,\ x_{%d})$" % n_feat,
                     "Valores normalizados con StandardScaler (0 = valor medio). La entrada no calcula nada:",
                     "solo entrega los valores a cada neurona de la capa siguiente."])
        if s < len(capas) - 1:
            L, Lp = capas[s], capas[s - 1]
            j, calc = ejemplo_calculo(s)
            act = int((L["a"] > 0).sum())
            var = "x_i" if s == 1 else "a_i"
            return (f"Paso {s + 1}/{n_pasos} · CAPA OCULTA {s}: {Lp['n']} → {L['n']} neuronas (ReLU)",
                    [f"Cada neurona recibe las {Lp['n']} salidas anteriores, las multiplica por sus pesos, "
                     "suma un sesgo y aplica ReLU.",
                     r"$z = \sum_i %s\,w_i + b \qquad a = \mathrm{ReLU}(z) = \max(0,\ z)$" % var,
                     f"Neurona {j + 1} de {L['n']} (naranja):  {calc}  →  a = {L['a'][j]:.2f}",
                     f"Activas: {act} de {L['n']} (ReLU apagó {L['n'] - act}, a = 0).  Keras lo calcula para "
                     f"todas a la vez: {Lp['n'] * L['n']} pesos + {L['n']} sesgos."])
        if s == len(capas) - 1:
            L = capas[s]
            j, calc = ejemplo_calculo(s)
            return (f"Paso {s + 1}/{n_pasos} · CAPA DE SALIDA: {capas[s - 1]['n']} → 1 neurona (Sigmoide)",
                    [f"La neurona de salida combina las {capas[s - 1]['n']} activaciones anteriores y aplica Sigmoide.",
                     r"$z = \sum_i a_i\,w_i + b \qquad p = \sigma(z) = \dfrac{1}{1 + e^{-z}}$",
                     f"{calc}  →  p = σ({L['z'][0]:.2f}) = {p_manual:.3f}",
                     "La sigmoide transforma cualquier número real en una probabilidad entre 0 y 1."])
        nom = {1: "VINO BUENO", 0: "VINO NO BUENO"}
        acierto = "CORRECTA ✓" if clase_pred == clase_real else "INCORRECTA ✗"
        return (f"Paso {n_pasos}/{n_pasos} · RESULTADO obtenido con {nombre_modelo}.predict()",
                [f"Probabilidad de vino bueno: {p_predict:.2f}",
                 f"Umbral {umbral}:  p {'≥' if clase_pred else '<'} {umbral}  →  Clasificación final: {nom[clase_pred]}",
                 f"Clase real: {'BUENO' if clase_real else 'NO BUENO'}  →  predicción {acierto}",
                 f"Los mismos cálculos que hicimos a mano, repetidos por Keras en "
                 f"{' + '.join(str(L['n']) for L in capas[1:])} = {sum(L['n'] for L in capas[1:])} neuronas."])

    # ---------- 8) Paneles inferiores ----------
    def dibujar_funcion(s, g):
        ax_fn.cla()
        ax_fn.tick_params(labelsize=7)
        for lado in ("top", "right"):
            ax_fn.spines[lado].set_visible(False)
        if s == 0:
            ax_fn.axis("off")
            ax_fn.text(0.5, 0.5, "Entrada:\nsin función\nde activación", ha="center", va="center",
                       fontsize=10, color="#777777", transform=ax_fn.transAxes)
            return
        k = min(s, len(capas) - 1)
        L = capas[k]
        z = L["z"][L["ej"]]
        sig = L["tipo"] == "sigmoid"
        R = max(6.0 if sig else 3.0, abs(z) * 1.25)
        zz = np.linspace(-R, R, 200)
        ax_fn.plot(zz, 1 / (1 + np.exp(-zz)) if sig else np.maximum(0, zz), color="#444444", lw=1.8)
        ax_fn.axhline(0, color="#bbbbbb", lw=0.8); ax_fn.axvline(0, color="#bbbbbb", lw=0.8)
        if sig:
            ax_fn.axhline(umbral, color=C_ANAR, lw=0.8, ls="--")
            ax_fn.set_ylim(-0.08, 1.08)
        ax_fn.set_title("Sigmoide" if sig else "ReLU (neurona naranja)", fontsize=9)
        ax_fn.set_xlabel("z", fontsize=8, labelpad=1)
        if g > 0:
            ax_fn.plot([z], [L["a"][L["ej"]]], "o", color=C_ANAR, ms=8, mec="black", zorder=5)

    def dibujar_resultado(s, g):
        ax_res.cla(); ax_res.set_xlim(0, 1); ax_res.set_ylim(0, 1); ax_res.axis("off")
        ax_res.text(0.5, 0.95, "Probabilidad de vino bueno", ha="center", fontsize=9.5, weight="bold")
        ax_res.add_patch(Rectangle((0, 0.55), 1, 0.18, facecolor="#eeeeee", edgecolor="#999999"))
        ax_res.plot([umbral, umbral], [0.5, 0.78], color="black", lw=1.2, ls="--")
        ax_res.text(umbral, 0.44, f"umbral {umbral}", ha="center", fontsize=7.5)
        ax_res.text(0, 0.44, "0", ha="center", fontsize=7.5); ax_res.text(1, 0.44, "1", ha="center", fontsize=7.5)
        if s < n_pasos - 1:
            ax_res.text(0.5, 0.82, "p = ?", ha="center", fontsize=11, color="#888888")
            return
        color = "#1a9850" if clase_pred else "#d73027"
        ax_res.add_patch(Rectangle((0, 0.55), p_predict * g, 0.18, facecolor=color, edgecolor="none"))
        ax_res.text(0.5, 0.82, f"p = {p_predict * g:.2f}", ha="center", fontsize=11, weight="bold")
        if g >= 1:
            ax_res.text(0.5, 0.25, "VINO BUENO" if clase_pred else "VINO NO BUENO", ha="center",
                        fontsize=12.5, weight="bold", color=color)
            ok = clase_pred == clase_real
            ax_res.text(0.5, 0.07, f"Real: {'BUENO' if clase_real else 'NO BUENO'}  {'✓' if ok else '✗'}",
                        ha="center", fontsize=10, color="#1a9850" if ok else "#d73027")

    # ---------- 9) Actualización de cada frame ----------
    N_FLUJO = frames_por_paso // 2
    def actualizar(f):
        s = min(f // frames_por_paso, n_pasos - 1)          # el resultado dura el doble
        loc = f - s * frames_por_paso
        t = min(1.0, (loc + 1) / N_FLUJO)                                   # avance de las conexiones
        final = s == n_pasos - 1
        if s == 0:
            g = min(1.0, (loc + 1) / 4)                                     # aparición de la entrada
        elif final:
            g = min(1.0, (loc + 1) / 6)                                     # llenado de la barra
        else:
            g = min(1.0, max(0.0, (loc - N_FLUJO + 2) / 3))                 # activación de la capa

        titulo_paso, lineas = textos_paso(s)
        subtitulo.set_text(titulo_paso)
        for txt, contenido, tam in zip(linea, lineas, [14 if final else 11.5, 13, 10, 9.5]):
            txt.set_text(contenido)
            ajustar(txt, tam, ancho_txt)

        col = min(s, len(capas) - 1)
        franja.set_x(XS[col] - (0.215 if col == 0 else 0.055))
        franja.set_width(0.27 if col == 0 else 0.11)
        for k, cab in enumerate(cabeceras):
            cab.set_color("black" if k == col else "#999999")

        # Conexiones
        for k in range(1, len(capas)):
            segs, cn, signo = aportes[k - 1]
            if k < s or final:
                fr, alpha = 1.0, (0.35 if final else 0.15)
            elif k == s:
                fr, alpha = t, 0.9
            else:
                fr, alpha = 0.0, 0.0
            nuevos = segs.copy()
            nuevos[:, 1] = segs[:, 0] + fr * (segs[:, 1] - segs[:, 0])
            colores = np.array([to_rgba(C_POS if sg == 0 else C_NEG) for sg in signo])
            colores[:, 3] = alpha * (0.25 + 0.75 * cn)
            flujo[k - 1].set_segments(nuevos)
            flujo[k - 1].set_color(colores)
            base[k - 1].set_alpha(0.25 if k > s and not final else 0.6)

        # Neuronas
        for k, L in enumerate(capas):
            nv = len(L["vis"])
            vals = L["norm"][L["vis"]]
            if k > s:                                                       # todavía no calculada
                fc, ec = np.tile(GRIS_F, (nv, 1)), np.tile(GRIS_B, (nv, 1))
                sizes, lw = np.full(nv, 90.0), np.full(nv, 0.8)
                visibles_txt = False
            else:
                gk = g if k == s else 1.0
                if L["tipo"] == "entrada":
                    fc = cm_in(0.5 + 0.45 * vals * gk); sizes = 110 + 230 * np.abs(vals) * gk
                elif L["tipo"] == "sigmoid":
                    fc = cm_out(np.full(nv, 0.5 + (p_manual - 0.5) * gk)); sizes = np.full(nv, 650.0)
                else:
                    fc = cm_hid(0.08 + 0.85 * np.abs(vals) * gk); sizes = 80 + 340 * np.abs(vals) * gk
                ec = np.tile(np.array([0.15, 0.15, 0.15, 1]), (nv, 1))
                lw = np.full(nv, 1.8 if k == col else 0.8)
                atenuar = 1.0 if (k == col or final) else 0.45
                fc = fc.copy(); fc[:, 3] = atenuar; ec[:, 3] = atenuar
                visibles_txt = (k < s) or gk > 0.5
            nodos[k].set_facecolors(fc); nodos[k].set_edgecolors(ec)
            nodos[k].set_sizes(sizes); nodos[k].set_linewidths(lw)
            for txt, vi in zip(textos_val[k], L["vis"]):
                if not visibles_txt:
                    txt.set_text("")
                    continue
                if L["tipo"] == "entrada":
                    txt.set_text(f"{L['a'][vi]:+.2f}")
                elif L["tipo"] == "sigmoid":
                    txt.set_text(f"p = {L['a'][vi]:.2f}")
                else:
                    txt.set_text(f"{L['a'][vi]:.2f}")
                txt.set_alpha(1.0 if (k == col or final) else 0.5)

        # Anillo naranja en la neurona de ejemplo de la capa actual
        if 1 <= s < len(capas):
            L = capas[s]
            pos = L["pos"][list(L["vis"]).index(L["ej"])]
            anillo.set_offsets([pos]); anillo.set_sizes([900 if s == len(capas) - 1 else 520])
            anillo.set_visible(g > 0)
        else:
            anillo.set_visible(False)

        dibujar_funcion(s, g)
        dibujar_resultado(s, g)
        return []

    anim = animation.FuncAnimation(fig, actualizar, frames=(n_pasos + 1) * frames_por_paso,
                                   interval=intervalo_ms, repeat=True)
    plt.close(fig)        # evita que Colab muestre además una imagen estática

    print(f"Ejemplo: fila {indice} de X_test  ->  criterio: {criterio}")
    print(f"Activaciones obtenidas con: {metodo}")
    print(f"{nombre_modelo}.predict(): p = {p_predict:.4f}  |  última activación intermedia: {p_manual:.4f}  "
          f"|  diferencia: {abs(p_predict - p_manual):.1e}")
    print(f"Clasificación: {'VINO BUENO' if clase_pred else 'VINO NO BUENO'}  |  "
          f"Clase real: {'BUENO' if clase_real else 'NO BUENO'}")
    with plt.rc_context({"animation.embed_limit": 60}):   # MB; solo para esta animación
        try:   # frames JPEG: HTML ~4 veces más liviano que PNG
            with plt.rc_context({"animation.frame_format": "jpeg"}):
                return HTML(anim.to_jshtml(default_mode="loop"))
        except Exception:
            return HTML(anim.to_jshtml(default_mode="loop"))


# Otro ejemplo: indice=15   |   un vino NO BUENO bien clasificado: clase_preferida=0
display(animar_forward_propagation(modelo_2, X_test, y_test))
