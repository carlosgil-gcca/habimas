"""Interfaz local de HabiMAS para demo determinista y agentes con Ollama."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from habimas import (
    BASE,
    MODELOS_OLLAMA,
    MODELO_OLLAMA_DEFAULT,
    OLLAMA_BASE_URL_DEFAULT,
    crear_directorio_ejecucion,
    ejecutar_crew_local,
    ejecutar_demo_con_evidencia,
    leer_json,
    validar_perfil,
)


def porcentaje(valor: float) -> float:
    return round(valor * 100, 4)


CLAVES_PRIORIDAD = ("transporte", "servicios", "seguridad", "accesibilidad")


def rebalancear_prioridades(cambiada: str) -> None:
    """Reparte el porcentaje restante y mantiene la suma territorial en 100."""
    restantes = [clave for clave in CLAVES_PRIORIDAD if clave != cambiada]
    disponible = 100 - int(st.session_state[cambiada])
    valores_anteriores = [int(st.session_state[clave]) for clave in restantes]
    suma_anterior = sum(valores_anteriores)
    if suma_anterior == 0:
        nuevos = [disponible // len(restantes)] * len(restantes)
    else:
        nuevos = [round(disponible * valor / suma_anterior) for valor in valores_anteriores]
    nuevos[-1] += disponible - sum(nuevos)
    for clave, valor in zip(restantes, nuevos):
        st.session_state[clave] = max(0, min(100, valor))


def construir_perfil_formulario(base: dict, valores: dict) -> dict:
    """Traduce porcentajes de UI a las proporciones que consume el motor."""
    perfil = dict(base)
    perfil.update({
        "ingreso_mensual_neto": valores["ingreso"],
        "deudas_mensuales": valores["deudas"],
        "ahorro_disponible": valores["ahorro"],
        "reserva_no_utilizable": valores["reserva"],
        "habitaciones_minimas": valores["habitaciones"],
        "proporcion_ingreso_para_hipoteca_y_deudas": valores["proporcion_pago"] / 100,
        "plazo_anios": valores["plazo"],
        "tasa_anual_fija_simulada": valores["tasa"] / 100,
        "enganche_minimo_pct": valores["enganche"] / 100,
        "costos_compra_pct_simulado": valores["costos"] / 100,
        "pesos_zona": {
            "transporte": valores["transporte"] / 100,
            "servicios": valores["servicios"] / 100,
            "seguridad": valores["seguridad"] / 100,
            "accesibilidad": valores["accesibilidad"] / 100,
        },
        "peso_zona_final": valores["peso_zona"] / 100,
        "peso_financiero_final": (100 - valores["peso_zona"]) / 100,
    })
    return validar_perfil(perfil)


def mostrar_resultados(resultado: dict) -> None:
    ranking = resultado["ranking"]
    resumen = ranking["resumen_perfil"]
    st.success(f"Ejecución terminada. Evidencia: `{resultado['salida']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Efectivo para compra", f"${resumen['efectivo_para_compra_mxn']:,.0f}")
    c2.metric("Pago hipotecario máximo", f"${resumen['pago_hipoteca_maximo_mxn']:,.0f}/mes")
    c3.metric("Habitaciones mínimas", resumen["habitaciones_minimas"])

    filas = []
    for vivienda in ranking["ranking"]:
        filas.append({
            "Posición": str(vivienda.get("posicion", "Excluida")),
            "Vivienda": vivienda["nombre"],
            "Precio": vivienda["precio_mxn"],
            "Mensualidad": vivienda["mensualidad_mxn"],
            "Zona": vivienda["puntaje_zona"],
            "Finanzas": vivienda["puntaje_financiero"],
            "Puntaje final": vivienda["puntaje_final"],
            "Viable": vivienda["viable"],
        })
    tabla = pd.DataFrame(filas)
    st.subheader("Ranking validado")
    st.dataframe(
        tabla.style.format({"Precio": "${:,.0f}", "Mensualidad": "${:,.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Comparación de puntuaciones")
    grafica = tabla.set_index("Vivienda")[["Zona", "Finanzas", "Puntaje final"]].fillna(0)
    st.bar_chart(grafica)

    st.subheader("Mensualidad frente al límite")
    pagos = tabla.set_index("Vivienda")[["Mensualidad"]]
    pagos["Pago máximo"] = resumen["pago_hipoteca_maximo_mxn"]
    st.bar_chart(pagos)

    excluidas = [v for v in ranking["ranking"] if not v["viable"]]
    if excluidas:
        st.subheader("Opciones excluidas")
        for vivienda in excluidas:
            st.warning(f"**{vivienda['nombre']}**: {', '.join(vivienda['motivos_exclusion'])}")

    if resultado.get("modo") == "crew_local":
        st.subheader("Explicación de los agentes locales")
        st.markdown(resultado["explicacion"])
    else:
        with st.expander("Traza de la demostración"):
            st.code(resultado["explicacion"])

    st.caption(ranking["advertencia"])


def mostrar_interaccion_agentes(resultado: dict | None = None) -> None:
    """Explica la arquitectura y muestra evidencia observable de la última corrida."""
    st.subheader("Flujo multiagente de HabiMAS")
    st.write(
        "Los agentes trabajan en secuencia. Cada agente consulta una herramienta Python "
        "determinista y entrega su resultado al siguiente componente."
    )
    st.code(
        "Usuario y perfil\n"
        "       ↓\n"
        "T1 · Agente de perfil\n"
        "       ├──────────────→ T2 · Agente territorial\n"
        "       └──────────────→ T3 · Agente financiero\n"
        "                              ↓\n"
        "                   T4 · Agente coordinador\n"
        "                              ↓\n"
        "              Ranking verificado + explicación",
        language=None,
    )

    agentes = [
        (
            "T1 · Analista de perfil",
            "Perfil económico, restricciones y prioridades",
            "normalizar_perfil_usuario",
            "Perfil validado, efectivo disponible y pago máximo",
        ),
        (
            "T2 · Evaluación territorial",
            "Perfil normalizado y puntuaciones sintéticas de las viviendas",
            "puntuar_alternativas_de_zona",
            "Puntaje territorial ponderado para cada alternativa",
        ),
        (
            "T3 · Viabilidad financiera",
            "Perfil normalizado, precio, ahorro, tasa y plazo",
            "calcular_escenarios_hipotecarios",
            "Enganche, préstamo, mensualidad, viabilidad y exclusiones",
        ),
        (
            "T4 · Coordinador",
            "Entregables de T1, T2 y T3",
            "ranking_verificado",
            "Explicación final contrastada con el motor determinista",
        ),
    ]
    columnas = st.columns(4)
    for columna, (nombre, entrada, herramienta, salida) in zip(columnas, agentes):
        with columna:
            st.markdown(f"#### {nombre}")
            st.markdown(f"**Entrada**  \n{entrada}")
            st.markdown(f"**Herramienta**  \n`{herramienta}`")
            st.markdown(f"**Salida**  \n{salida}")

    st.divider()
    st.subheader("Control de confiabilidad")
    st.markdown(
        "1. El modelo local coordina las tareas y redacta explicaciones.  \n"
        "2. Las herramientas Python calculan todas las cifras.  \n"
        "3. El coordinador consulta el ranking autoritativo.  \n"
        "4. HabiMAS valida el informe antes de presentarlo.  \n"
        "5. Si sólo faltan exclusiones, las completa; si hay contradicción, sustituye el informe."
    )

    if resultado is None:
        st.info("Genere una recomendación para consultar la interacción de la última ejecución.")
        return

    st.divider()
    st.subheader("Última ejecución")
    modo = resultado.get("modo", "desconocido")
    salida = Path(resultado["salida"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Modo", "Agentes locales" if modo == "crew_local" else "Demo simulada")
    c2.metric("Etapas", "4")
    c3.metric("Evidencia", salida.name)

    ejecucion_path = salida / "ejecucion_agentes.json"
    if modo == "crew_local" and ejecucion_path.exists():
        ejecucion = json.loads(ejecucion_path.read_text(encoding="utf-8"))
        completadas = ejecucion["tareas_completadas"]
        configuradas = ejecucion["agentes_configurados"]
        if completadas == configuradas:
            st.success(
                f"Ejecución multiagente confirmada: {completadas}/{configuradas} tareas "
                f"completadas con {ejecucion['modelo']} mediante {ejecucion['framework']}."
            )
        else:
            st.warning(
                f"Ejecución incompleta: {completadas}/{configuradas} tareas terminadas."
            )
    elif modo == "crew_local":
        st.warning(
            "Esta evidencia fue creada por una versión anterior de la aplicación. "
            "Reinicie Streamlit y genere una ejecución nueva para obtener el detalle 4/4."
        )

    interacciones_path = salida / "interaccion_agentes.json"
    if interacciones_path.exists():
        interacciones = json.loads(interacciones_path.read_text(encoding="utf-8"))
        st.subheader("Entrada y salida de cada agente")
        st.write(
            "Cada bloque separa los datos recibidos, la operación verificable y el texto "
            "que el agente entregó al siguiente paso."
        )
        for indice, interaccion in enumerate(interacciones):
            titulo = f"{interaccion['tarea']} · {interaccion['agente']}"
            with st.expander(titulo, expanded=indice == 0):
                st.markdown(f"**Objetivo**  \n{interaccion['objetivo']}")
                st.markdown("**1. Entrada recibida**")
                st.json(interaccion["entrada"], expanded=1)
                st.markdown(
                    f"**2. Herramienta invocada**  \n`{interaccion['herramienta']}`"
                )
                st.markdown("**3. Operaciones ejecutadas por Python**")
                for paso in interaccion["proceso_determinista"]:
                    st.markdown(f"- {paso}")
                st.markdown("**4. Salida verificable de la herramienta**")
                st.json(interaccion["salida_herramienta"], expanded=1)
                etiqueta = (
                    "Entregable real producido por CrewAI"
                    if interaccion["tipo_entregable"] == "real_crewai"
                    else "Entregable simulado"
                )
                st.markdown(f"**5. {etiqueta}**")
                st.code(interaccion["entregable_agente"], language=None)

    if modo == "demo":
        st.warning(
            "Esta ejecución no utilizó un LLM: reproduce el mismo flujo T1–T4 de forma simulada."
        )
        with st.expander("Ver traza simulada", expanded=True):
            st.code(resultado["explicacion"])
        return

    validacion_path = salida / "validacion_informe.json"
    if validacion_path.exists():
        validacion = json.loads(validacion_path.read_text(encoding="utf-8"))
        accion = validacion.get("accion", "sin información")
        if accion == "aceptado":
            st.success("El informe del Crew coincidió con el ranking verificado.")
        elif accion == "completado_con_exclusiones_deterministas":
            st.info("El informe del Crew fue conservado y se completaron sus exclusiones.")
        else:
            st.warning("El informe bruto se conservó, pero la presentación requirió sustitución.")

    log_path = salida / "log_crew_entregables.txt"
    if log_path.exists():
        with st.expander("Entregables reales entre agentes", expanded=True):
            st.code(log_path.read_text(encoding="utf-8"), language=None)

    bruto_path = salida / "informe_crew_bruto.md"
    if bruto_path.exists():
        with st.expander("Informe bruto producido por el coordinador"):
            st.markdown(bruto_path.read_text(encoding="utf-8"))

    st.caption(
        "Se muestran entregables y llamadas observables, no razonamientos privados del modelo."
    )


st.set_page_config(page_title="HabiMAS", page_icon="🏠", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1450px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #fff8f3 0%, #ffffff 100%);
        border: 1px solid #f0d7c4; border-radius: 14px; padding: 14px 16px;
    }
    [data-testid="stExpander"] {border: 1px solid #eadfd7; border-radius: 12px;}
    .habimas-hero {
        padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #7a3218 0%, #c65d2e 58%, #e98b52 100%);
        color: white; box-shadow: 0 8px 22px rgba(122, 50, 24, .16);
    }
    .habimas-hero h1 {margin: 0; color: white; font-size: 2.15rem;}
    .habimas-hero p {margin: .45rem 0 0; opacity: .94; font-size: 1.02rem;}
    .habimas-pill {
        display: inline-block; margin: .25rem .35rem .1rem 0; padding: .28rem .7rem;
        border-radius: 999px; background: rgba(255,255,255,.18); font-size: .84rem;
    }
    </style>
    <div class="habimas-hero">
      <h1>🏠 HabiMAS</h1>
      <p>Asistente multiagente para comparar alternativas de vivienda y viabilidad financiera.</p>
      <span class="habimas-pill">4 agentes especializados</span>
      <span class="habimas-pill">Modelo local con Ollama</span>
      <span class="habimas-pill">Ranking verificable</span>
    </div>
    """,
    unsafe_allow_html=True,
)

resumen_a, resumen_b, resumen_c = st.columns(3)
resumen_a.metric("Arquitectura", "4 agentes", "CrewAI secuencial")
resumen_b.metric("Privacidad", "Ejecución local", "Sin costo de API")
resumen_c.metric("Decisión", "Motor auditable", "Cálculos en Python")

st.caption(
    "Prototipo académico con viviendas, indicadores, tasas y costos simulados. "
    "El ranking numérico siempre proviene del motor determinista."
)

perfil_base = leer_json(BASE / "perfil_usuario.json")
viviendas = leer_json(BASE / "viviendas_simuladas.json")
for clave in CLAVES_PRIORIDAD:
    if clave not in st.session_state:
        st.session_state[clave] = int(porcentaje(perfil_base["pesos_zona"][clave]))

with st.sidebar:
    st.header("⚙️ Ejecución")
    modo_ui = st.radio("Modo", ("Demo determinista", "Agentes locales"))
    modelo = st.selectbox(
        "Modelo Ollama",
        MODELOS_OLLAMA,
        index=MODELOS_OLLAMA.index(MODELO_OLLAMA_DEFAULT),
        disabled=modo_ui == "Demo determinista",
    )
    ollama_url = st.text_input(
        "URL de Ollama", OLLAMA_BASE_URL_DEFAULT,
        disabled=modo_ui == "Demo determinista",
    )
    st.divider()
    st.caption(
        "Use Demo para una respuesta inmediata o Agentes locales para ejecutar el flujo T1–T4."
    )

tab_configuracion, tab_recomendacion, tab_agentes = st.tabs(
    ("⚙️ Configuración", "📊 Recomendación", "🤖 Interacción de agentes")
)

with tab_configuracion:
    st.subheader("1. Perfil financiero")
    st.write("Defina la capacidad económica y las restricciones obligatorias del caso.")
    a, b, c, d = st.columns(4)
    ingreso = a.number_input("Ingreso mensual neto", min_value=1.0, value=float(perfil_base["ingreso_mensual_neto"]), step=1000.0)
    deudas = b.number_input("Deudas mensuales", min_value=0.0, value=float(perfil_base["deudas_mensuales"]), step=500.0)
    ahorro = c.number_input("Ahorro disponible", min_value=0.0, value=float(perfil_base["ahorro_disponible"]), step=10000.0)
    reserva = d.number_input("Reserva no utilizable", min_value=0.0, value=float(perfil_base["reserva_no_utilizable"]), step=10000.0)

    e, f, g, h, i, j = st.columns(6)
    habitaciones = e.number_input("Habitaciones mínimas", min_value=1, value=int(perfil_base["habitaciones_minimas"]), step=1)
    plazo = f.number_input("Plazo (años)", min_value=1, value=int(perfil_base["plazo_anios"]), step=1)
    tasa = g.number_input("Tasa anual (%)", min_value=0.0, max_value=99.0, value=porcentaje(perfil_base["tasa_anual_fija_simulada"]), step=0.1)
    proporcion_pago = h.number_input("Límite ingreso (%)", min_value=0.0, max_value=99.0, value=porcentaje(perfil_base["proporcion_ingreso_para_hipoteca_y_deudas"]), step=1.0)
    enganche = i.number_input("Enganche mínimo (%)", min_value=0.0, max_value=99.0, value=porcentaje(perfil_base["enganche_minimo_pct"]), step=1.0)
    costos = j.number_input("Costos compra (%)", min_value=0.0, max_value=99.0, value=porcentaje(perfil_base["costos_compra_pct_simulado"]), step=1.0)

    st.divider()
    st.subheader("2. Prioridades territoriales")
    st.write(
        "Distribuya 100 puntos entre los cuatro criterios. Al mover un control, "
        "los otros tres se ajustan automáticamente."
    )
    k, l, m, n = st.columns(4)
    transporte = k.slider(
        "Transporte (%)", 0, 100, key="transporte",
        on_change=rebalancear_prioridades, args=("transporte",),
    )
    servicios = l.slider(
        "Servicios (%)", 0, 100, key="servicios",
        on_change=rebalancear_prioridades, args=("servicios",),
    )
    seguridad = m.slider(
        "Seguridad (%)", 0, 100, key="seguridad",
        on_change=rebalancear_prioridades, args=("seguridad",),
    )
    accesibilidad = n.slider(
        "Accesibilidad (%)", 0, 100, key="accesibilidad",
        on_change=rebalancear_prioridades, args=("accesibilidad",),
    )
    suma = transporte + servicios + seguridad + accesibilidad
    st.progress(suma / 100, text=f"Total asignado: {suma}% de 100%")
    st.divider()
    st.subheader("3. Balance del ranking")
    peso_zona = st.slider(
        "Peso territorial en el ranking final (%)", 0, 100,
        int(porcentaje(perfil_base["peso_zona_final"])),
        help="El porcentaje restante se asigna automáticamente al componente financiero.",
    )
    st.caption(
        f"Composición final: {peso_zona}% territorio · {100 - peso_zona}% finanzas"
    )
    ejecutar = st.button("Generar recomendación", type="primary")

if ejecutar:
    valores = locals().copy()
    try:
        perfil = construir_perfil_formulario(perfil_base, valores)
        if modo_ui == "Demo determinista":
            salida = crear_directorio_ejecucion("demo")
            resultado = ejecutar_demo_con_evidencia(perfil, viviendas, salida)
            resultado["modo"] = "demo"
        else:
            salida = crear_directorio_ejecucion("crew_local", modelo)
            with st.spinner(f"Ejecutando cuatro agentes con {modelo}..."):
                resultado = ejecutar_crew_local(perfil, viviendas, salida, modelo, ollama_url)
            resultado["modo"] = "crew_local"
        st.session_state["resultado_habimas"] = resultado
    except Exception as exc:
        st.error(str(exc))

resultado_actual = st.session_state.get("resultado_habimas")
with tab_recomendacion:
    if resultado_actual:
        mostrar_resultados(resultado_actual)
    else:
        st.info("Configure el perfil y presione “Generar recomendación” para ver resultados.")
with tab_agentes:
    mostrar_interaccion_agentes(resultado_actual)
