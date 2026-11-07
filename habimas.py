"""HabiMAS - Prototipo académico de recomendación de vivienda.

Dos modos:
    python habimas.py --modo demo   # reproducible, SIN LLM / SIN CrewAI
    python habimas.py --modo crew   # CrewAI + proveedor LLM configurado

IMPORTANTE: Datos, tasas, costos y puntuaciones de ejemplo son SIMULADOS.
El modo demo no ejecuta agentes LLM: genera un registro de tareas emulado.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
CAMPOS_ZONA = ("transporte", "servicios", "seguridad", "accesibilidad")
MODELOS_OLLAMA = ("llama3.1:8b", "llama3.2:latest")
MODELO_OLLAMA_DEFAULT = "llama3.1:8b"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434/v1"


def leer_json(ruta: Path) -> Any:
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_json(ruta: Path, contenido: Any) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")


def validar_modelo_local(modelo: str) -> str:
    """Limita la interfaz académica a modelos locales previamente evaluados."""
    if modelo not in MODELOS_OLLAMA:
        raise ValueError(
            f"Modelo local no admitido: {modelo}. Opciones: {', '.join(MODELOS_OLLAMA)}"
        )
    return modelo


def _url_ollama_raiz(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def consultar_modelos_ollama(base_url: str = OLLAMA_BASE_URL_DEFAULT, timeout: float = 3) -> set[str]:
    """Comprueba el servidor Ollama y devuelve los nombres de modelos instalados."""
    url = f"{_url_ollama_raiz(base_url)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as respuesta:
            contenido = json.loads(respuesta.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ConnectionError(
            "No fue posible conectar con Ollama. Inicia Ollama y confirma que "
            f"responda en {_url_ollama_raiz(base_url)}."
        ) from exc
    return {item.get("name", "") for item in contenido.get("models", [])}


def comprobar_ollama(modelo: str, base_url: str = OLLAMA_BASE_URL_DEFAULT) -> None:
    """Valida modelo, conectividad y disponibilidad antes de iniciar el Crew."""
    validar_modelo_local(modelo)
    instalados = consultar_modelos_ollama(base_url)
    if modelo not in instalados:
        raise RuntimeError(
            f"El modelo {modelo} no está instalado. Ejecuta: ollama pull {modelo}"
        )


def crear_directorio_ejecucion(
    modo: str, modelo: str | None = None, raiz: Path = BASE / "salidas" / "ejecuciones"
) -> Path:
    """Crea una carpeta única y legible para conservar evidencia de una corrida."""
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    etiqueta = modelo or modo
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", etiqueta).strip("-").lower()
    ruta = raiz / f"{marca}_{slug}_{uuid.uuid4().hex[:6]}"
    ruta.mkdir(parents=True, exist_ok=False)
    return ruta


def guardar_evidencia_base(
    salida: Path, perfil: dict, ranking: dict, modo: str, modelo: str | None = None
) -> None:
    guardar_json(salida / "perfil_utilizado.json", perfil)
    guardar_json(salida / "ranking_validado.json", ranking)
    guardar_json(salida / "configuracion.json", {
        "modo": modo,
        "modelo": modelo,
        "fecha_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


def construir_interacciones_agentes(
    perfil: dict,
    viviendas: list[dict],
    entregables: list[str] | None = None,
    modo: str = "demo",
) -> list[dict]:
    """Crea una traza auditable de entradas, herramientas, salidas y entregables."""
    perfil_validado = validar_perfil(perfil)
    zonas = puntuar_zonas(perfil, viviendas)
    finanzas = evaluar_finanzas(perfil, viviendas)
    ranking = construir_ranking(perfil, viviendas)
    entregables = entregables or []

    def entrega(indice: int, simulada: str) -> str:
        return entregables[indice] if indice < len(entregables) else simulada

    return [
        {
            "tarea": "T1",
            "agente": "Analista de Perfil del Usuario",
            "objetivo": "Validar y estructurar las restricciones económicas y preferencias.",
            "entrada": perfil,
            "herramienta": "normalizar_perfil_usuario",
            "proceso_determinista": [
                "Valida campos obligatorios y porcentajes.",
                "Resta la reserva al ahorro disponible.",
                "Calcula el límite mensual después de deudas.",
            ],
            "salida_herramienta": perfil_validado,
            "entregable_agente": entrega(
                0,
                "Simulación: perfil validado y restricciones transmitidas a T2 y T3.",
            ),
            "tipo_entregable": "real_crewai" if entregables else "simulado",
        },
        {
            "tarea": "T2",
            "agente": "Especialista en Evaluación Territorial",
            "objetivo": "Comparar las viviendas con las prioridades territoriales del usuario.",
            "entrada": {
                "pesos_zona": perfil_validado["pesos_zona"],
                "alternativas": [
                    {"id": v["id"], "puntajes_zona": v["puntajes_zona"]}
                    for v in viviendas
                ],
            },
            "herramienta": "puntuar_alternativas_de_zona",
            "proceso_determinista": [
                "Multiplica cada indicador por su prioridad.",
                "Suma los cuatro componentes.",
                "Entrega un puntaje territorial de 0 a 100.",
            ],
            "salida_herramienta": zonas,
            "entregable_agente": entrega(
                1,
                "Simulación: puntuaciones territoriales transmitidas al coordinador.",
            ),
            "tipo_entregable": "real_crewai" if entregables else "simulado",
        },
        {
            "tarea": "T3",
            "agente": "Analista de Viabilidad Financiera",
            "objetivo": "Calcular pagos y excluir alternativas que incumplen restricciones duras.",
            "entrada": {
                "capacidad": {
                    "efectivo_para_compra_mxn": perfil_validado["efectivo_para_compra_mxn"],
                    "pago_hipoteca_maximo_mxn": perfil_validado["pago_hipoteca_maximo_mxn"],
                    "plazo_anios": perfil_validado["plazo_anios"],
                    "tasa_anual": perfil_validado["tasa_anual_fija_simulada"],
                    "enganche_minimo_pct": perfil_validado["enganche_minimo_pct"],
                },
                "alternativas": [
                    {
                        "id": v["id"], "precio_mxn": v["precio_mxn"],
                        "habitaciones": v["habitaciones"],
                    }
                    for v in viviendas
                ],
            },
            "herramienta": "calcular_escenarios_hipotecarios",
            "proceso_determinista": [
                "Calcula gastos iniciales, enganche disponible y préstamo.",
                "Aplica amortización francesa para obtener la mensualidad.",
                "Evalúa liquidez, habitaciones y capacidad mensual.",
            ],
            "salida_herramienta": finanzas,
            "entregable_agente": entrega(
                2,
                "Simulación: escenarios financieros transmitidos al coordinador.",
            ),
            "tipo_entregable": "real_crewai" if entregables else "simulado",
        },
        {
            "tarea": "T4",
            "agente": "Coordinador de Recomendaciones de Vivienda",
            "objetivo": "Integrar evaluaciones, aplicar exclusiones y explicar el ranking final.",
            "entrada": {
                "perfil_normalizado": perfil_validado,
                "puntuaciones_territoriales": zonas,
                "escenarios_financieros": finanzas,
            },
            "herramienta": "ranking_verificado",
            "proceso_determinista": [
                "Excluye primero todas las opciones no viables.",
                "Combina zona y confort financiero con los pesos elegidos.",
                "Ordena alternativas viables y conserva motivos de exclusión.",
            ],
            "salida_herramienta": ranking,
            "entregable_agente": entrega(
                3,
                "Simulación: ranking y advertencias preparados para la interfaz.",
            ),
            "tipo_entregable": "real_crewai" if entregables else "simulado",
        },
    ]


def validar_perfil(perfil: dict) -> dict:
    """Aplica comprobaciones deterministas antes de consultar el LLM."""
    obligatorios = (
        "ingreso_mensual_neto", "deudas_mensuales", "ahorro_disponible",
        "reserva_no_utilizable", "habitaciones_minimas",
        "proporcion_ingreso_para_hipoteca_y_deudas", "plazo_anios",
        "tasa_anual_fija_simulada", "enganche_minimo_pct",
        "costos_compra_pct_simulado", "pesos_zona", "peso_zona_final",
        "peso_financiero_final",
    )
    faltantes = [campo for campo in obligatorios if campo not in perfil]
    if faltantes:
        raise ValueError(f"Faltan campos del perfil: {faltantes}")
    if perfil["ingreso_mensual_neto"] <= 0:
        raise ValueError("El ingreso debe ser positivo")
    for clave in ("deudas_mensuales", "ahorro_disponible", "reserva_no_utilizable"):
        if perfil[clave] < 0:
            raise ValueError(f"El campo {clave} no puede ser negativo")
    if perfil["reserva_no_utilizable"] > perfil["ahorro_disponible"]:
        raise ValueError("La reserva supera el ahorro disponible")
    if perfil["habitaciones_minimas"] < 1 or perfil["plazo_anios"] <= 0:
        raise ValueError("Habitaciones y plazo deben ser positivos")
    for campo in (
        "proporcion_ingreso_para_hipoteca_y_deudas", "enganche_minimo_pct",
        "costos_compra_pct_simulado", "tasa_anual_fija_simulada"
    ):
        if not (0 <= perfil[campo] < 1):
            raise ValueError(f"Porcentaje fuera de rango: {campo}")
    pesos = perfil["pesos_zona"]
    if set(pesos) != set(CAMPOS_ZONA) or any(pesos[c] < 0 for c in CAMPOS_ZONA):
        raise ValueError("Los pesos de zona son incompletos o inválidos")
    if abs(sum(pesos.values()) - 1) > 1e-8:
        raise ValueError("Los pesos de zona deben sumar 1")
    if abs(perfil["peso_zona_final"] + perfil["peso_financiero_final"] - 1) > 1e-8:
        raise ValueError("Los pesos del ranking deben sumar 1")
    if min(perfil["peso_zona_final"], perfil["peso_financiero_final"]) < 0:
        raise ValueError("Los pesos del ranking no pueden ser negativos")
    salida = dict(perfil)
    salida["efectivo_para_compra_mxn"] = round(
        perfil["ahorro_disponible"] - perfil["reserva_no_utilizable"], 2
    )
    salida["pago_hipoteca_maximo_mxn"] = round(max(
        0, perfil["ingreso_mensual_neto"]
        * perfil["proporcion_ingreso_para_hipoteca_y_deudas"]
        - perfil["deudas_mensuales"]
    ), 2)
    return salida


def validar_viviendas(viviendas: list[dict]) -> None:
    """Rechaza datos incompletos y duplicaciones de IDs."""
    if not viviendas or len({v.get('id') for v in viviendas}) != len(viviendas):
        raise ValueError("Se requieren viviendas con IDs distintos")
    for v in viviendas:
        if not v.get("id") or not v.get("nombre") or v["precio_mxn"] <= 0:
            raise ValueError("La vivienda debe tener ID, nombre y precio positivo")
        if v["habitaciones"] < 0:
            raise ValueError("Habitaciones inválidas")
        valores = v["puntajes_zona"]
        if set(valores) != set(CAMPOS_ZONA):
            raise ValueError(f"Faltan criterios en {v['id']}")
        if any(not (0 <= valores[k] <= 100) for k in CAMPOS_ZONA):
            raise ValueError(f"Los puntajes deben estar entre 0 y 100: {v['id']}")


def puntuar_zonas(perfil: dict, viviendas: list[dict]) -> list[dict]:
    """Suma ponderada 0-100; no inventa indicadores de colonias reales."""
    p = validar_perfil(perfil)
    validar_viviendas(viviendas)
    return [
        {
            "id": v["id"],
            "puntaje_zona": round(sum(
                v["puntajes_zona"][criterio] * p["pesos_zona"][criterio]
                for criterio in CAMPOS_ZONA
            ), 2),
            "criterios": v["puntajes_zona"],
        }
        for v in viviendas
    ]


def mensualidad_francesa(prestamo: float, tasa_anual: float, plazo_anios: int) -> float:
    """Pago mensual nivelado de capital + interés, SIN seguros ni comisiones."""
    if prestamo <= 0:
        return 0.0
    meses = plazo_anios * 12
    tasa_mensual = tasa_anual / 12
    if tasa_mensual == 0:
        return prestamo / meses
    return prestamo * tasa_mensual / (1 - (1 + tasa_mensual) ** (-meses))


def evaluar_finanzas(perfil: dict, viviendas: list[dict]) -> list[dict]:
    """Evalúa restricciones duras (enganche, liquidez, pago y habitaciones)."""
    p = validar_perfil(perfil)
    validar_viviendas(viviendas)
    resultados = []
    for v in viviendas:
        precio = v["precio_mxn"]
        gastos = precio * p["costos_compra_pct_simulado"]
        efectivo = p["efectivo_para_compra_mxn"]
        enganche_maximo = max(0.0, efectivo - gastos)
        enganche = min(precio, enganche_maximo)
        enganche_minimo = precio * p["enganche_minimo_pct"]
        prestamo = max(0.0, precio - enganche)
        mensualidad = mensualidad_francesa(
            prestamo, p["tasa_anual_fija_simulada"], p["plazo_anios"]
        )
        motivos = []
        if enganche + 1e-7 < enganche_minimo:
            motivos.append("Enganche insuficiente")
        if gastos + enganche_minimo > efectivo + 1e-7:
            motivos.append("Liquidez inicial insuficiente")
        if v["habitaciones"] < p["habitaciones_minimas"]:
            motivos.append("Habitaciones insuficientes")
        if mensualidad > p["pago_hipoteca_maximo_mxn"] + 1e-7:
            motivos.append("Mensualidad supera capacidad de pago")
        viable = len(motivos) == 0
        capacidad = p["pago_hipoteca_maximo_mxn"]
        # El confort financiero mide cuánto margen queda para imprevistos.
        puntaje_financiero = (
            round(100 * max(0.0, 1 - mensualidad / capacidad), 2)
            if viable and capacidad > 0 else 0.0
        )
        resultados.append({
            "id": v["id"],
            "precio_mxn": precio,
            "gastos_compra_mxn": round(gastos, 2),
            "enganche_mxn": round(enganche, 2),
            "prestamo_mxn": round(prestamo, 2),
            "mensualidad_mxn": round(mensualidad, 2),
            "pago_maximo_mxn": capacidad,
            "puntaje_financiero": puntaje_financiero,
            "viable": viable,
            "motivos_exclusion": motivos,
        })
    return resultados


def construir_ranking(perfil: dict, viviendas: list[dict]) -> dict:
    """Regla de decisión auditable: exclusión dura + ranking ponderado."""
    p = validar_perfil(perfil)
    zona = {x["id"]: x for x in puntuar_zonas(perfil, viviendas)}
    fin = {x["id"]: x for x in evaluar_finanzas(perfil, viviendas)}
    opciones = []
    for v in viviendas:
        zid, fid = zona[v["id"]], fin[v["id"]]
        puntaje = round(
            p["peso_zona_final"] * zid["puntaje_zona"]
            + p["peso_financiero_final"] * fid["puntaje_financiero"], 2
        ) if fid["viable"] else None
        opciones.append({
            "id": v["id"], "nombre": v["nombre"],
            "habitaciones": v["habitaciones"],
            "puntaje_zona": zid["puntaje_zona"],
            **fid, "puntaje_final": puntaje,
        })
    opciones.sort(key=lambda v: (not v["viable"], -(v["puntaje_final"] or 0), v["id"]))
    for pos, opcion in enumerate((v for v in opciones if v["viable"]), start=1):
        opcion["posicion"] = pos
    return {
        "tipo_ejecucion": "motor_determinista",
        "advertencia": "Precios, indicadores y tasa son simulados; no constituyen una oferta hipotecaria.",
        "resumen_perfil": {
            "efectivo_para_compra_mxn": p["efectivo_para_compra_mxn"],
            "pago_hipoteca_maximo_mxn": p["pago_hipoteca_maximo_mxn"],
            "habitaciones_minimas": p["habitaciones_minimas"],
        },
        "regla": "Excluir restricciones duras; score=0.70*zona+0.30*confort_financiero",
        "ranking": opciones,
    }


def simulacion(perfil: dict, viviendas: list[dict]) -> tuple[dict, str]:
    """Registra un traspaso de tareas SIMULADO, no razonamiento de LLM."""
    p = validar_perfil(perfil)
    z = puntuar_zonas(perfil, viviendas)
    f = evaluar_finanzas(perfil, viviendas)
    resultado = construir_ranking(perfil, viviendas)
    trazas = [
        "=== DEMOSTRACION SIMULADA - NO EJECUTA CREWAI NI UN LLM ===",
        "[T1 | PERFIL] Validación correcta. Habitaciones mínimas: "
        f"{p['habitaciones_minimas']}; efectivo para compra: ${p['efectivo_para_compra_mxn']:,.0f}; "
        f"máximo hipoteca: ${p['pago_hipoteca_maximo_mxn']:,.0f}/mes.",
        "[CONTEXTO] El perfil normalizado se transmite a T2 y T3.",
        "[T2 | ZONA] Puntuaciones ponderadas: "
        + ", ".join(f"{x['id']}={x['puntaje_zona']}" for x in z),
        "[T3 | FINANZAS] Resultado de factibilidad: "
        + ", ".join(f"{x['id']}={'sí' if x['viable'] else 'no'} "
                    f"(${x['mensualidad_mxn']:,.2f}/mes)" for x in f),
        "[CONTEXTO] T4 recibe los resultados de T1, T2 y T3.",
        "[T4 | COORDINADOR] Se excluyen restricciones duras y se ordena el ranking.",
    ]
    for x in resultado["ranking"]:
        if x["viable"]:
            trazas.append(
                f"[RANK {x['posicion']}] {x['id']}: score={x['puntaje_final']}, "
                f"mensualidad=${x['mensualidad_mxn']:,.2f}"
            )
        else:
            trazas.append(f"[EXCLUIDA] {x['id']}: {', '.join(x['motivos_exclusion'])}")
    return resultado, "\n".join(trazas) + "\n"


def crear_llm_local(modelo: str, base_url: str = OLLAMA_BASE_URL_DEFAULT):
    """Construye un LLM de CrewAI sobre la API OpenAI-compatible de Ollama."""
    from crewai import LLM

    validar_modelo_local(modelo)
    return LLM(
        model=modelo,
        custom_openai=True,
        base_url=base_url.rstrip("/"),
        api_key="ollama",
        temperature=0,
    )


def crear_crew(
    perfil: dict,
    viviendas: list[dict],
    salida: Path,
    modelo: str = MODELO_OLLAMA_DEFAULT,
    base_url: str = OLLAMA_BASE_URL_DEFAULT,
):
    """Crea CUATRO agentes y CUATRO tareas CrewAI reales (requiere API key)."""
    # Importación diferida: el modo demo no necesita instalar CrewAI.
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool
    from pydantic import BaseModel, ConfigDict

    llm = crear_llm_local(modelo, base_url)

    class EntradaFlexible(BaseModel):
        """Tolera argumentos superfluos de modelos pequeños; las herramientas no los usan."""
        model_config = ConfigDict(extra="allow")

    class HerramientaPerfil(BaseTool):
        name: str = "normalizar_perfil_usuario"
        description: str = "Retorna el perfil real cargado y validado. No requiere argumentos."
        args_schema: type[BaseModel] = EntradaFlexible

        def _run(self, **kwargs: Any) -> str:
            return json.dumps(validar_perfil(perfil), ensure_ascii=False)

    class HerramientaZona(BaseTool):
        name: str = "puntuar_alternativas_de_zona"
        description: str = "Retorna puntuaciones calculadas de las viviendas. No requiere argumentos."
        args_schema: type[BaseModel] = EntradaFlexible

        def _run(self, **kwargs: Any) -> str:
            return json.dumps(puntuar_zonas(perfil, viviendas), ensure_ascii=False)

    class HerramientaFinanzas(BaseTool):
        name: str = "calcular_escenarios_hipotecarios"
        description: str = "Retorna cálculos financieros verificados. No requiere argumentos."
        args_schema: type[BaseModel] = EntradaFlexible

        def _run(self, **kwargs: Any) -> str:
            return json.dumps(evaluar_finanzas(perfil, viviendas), ensure_ascii=False)

    class HerramientaRanking(BaseTool):
        name: str = "ranking_verificado"
        description: str = "Retorna el ranking autoritativo. No requiere argumentos."
        args_schema: type[BaseModel] = EntradaFlexible

        def _run(self, **kwargs: Any) -> str:
            return json.dumps(construir_ranking(perfil, viviendas), ensure_ascii=False)

    herramienta_perfil = HerramientaPerfil()
    herramienta_zona = HerramientaZona()
    herramienta_finanzas = HerramientaFinanzas()
    herramienta_ranking = HerramientaRanking()

    perfil_agente = Agent(
        role="Analista de Perfil del Usuario",
        goal="Normalizar las necesidades, validar preferencias y distinguir restricciones obligatorias.",
        backstory="Especialista en levantamiento de requisitos de compradores de vivienda; "
                  "nunca supone datos faltantes ni modifica el presupuesto del usuario.",
        tools=[herramienta_perfil], llm=llm, allow_delegation=False,
        verbose=True, max_iter=5,
    )
    zona_agente = Agent(
        role="Especialista en Evaluación Territorial",
        goal="Comparar criterios de zona mediante un método ponderado verificable.",
        backstory="Analista urbano que diferencia indicadores objetivos de preferencias; "
                  "trabaja solamente con los datos simulados proporcionados.",
        tools=[herramienta_zona], llm=llm, allow_delegation=False,
        verbose=True, max_iter=5,
    )
    fin_agente = Agent(
        role="Analista de Viabilidad Financiera",
        goal="Detectar viviendas sin liquidez o capacidad mensual y calcular pagos exactos.",
        backstory="Analista de crédito prudente: no presenta simulaciones como cotizaciones "
                  "bancarias ni oculta comisiones o seguros no modelados.",
        tools=[herramienta_finanzas], llm=llm, allow_delegation=False,
        verbose=True, max_iter=5,
    )
    coord_agente = Agent(
        role="Coordinador de Recomendaciones de Vivienda",
        goal="Integrar hallazgos, excluir inviables y justificar un ranking consistente.",
        backstory="Responsable de decisiones multicriterio: prioriza restricciones duras "
                  "y coteja números con un motor determinista antes de explicar conclusiones.",
        tools=[herramienta_ranking], llm=llm, allow_delegation=False,
        verbose=True, max_iter=5,
    )

    # T1 genera contexto para T2, T3 y T4.
    t1 = Task(
        description="Invoca normalizar_perfil_usuario sin inventar entradas. Entrega el JSON devuelto "
                    "y explica en 2 frases cuáles son las restricciones duras. "
                    "No inventes ingresos ni gustos.",
        expected_output="JSON del perfil validado y resumen de restricciones.",
        agent=perfil_agente,
    )
    t2 = Task(
        description="Lee el perfil de T1 e invoca puntuar_alternativas_de_zona sin argumentos. "
                    "Devuelve ID y puntuación territorial de cada alternativa. "
                    "Nunca uses información no contenida en los datos de entrada.",
        expected_output="Lista JSON con ID, criterios y puntaje_zona 0-100.",
        agent=zona_agente, context=[t1],
    )
    t3 = Task(
        description="Lee el perfil de T1 e invoca calcular_escenarios_hipotecarios sin argumentos. "
                    "Entrega todas las opciones con pago, enganche, préstamo, "
                    "viabilidad y motivos de exclusión. Aclara que son estimaciones.",
        expected_output="Lista JSON de cálculo financiero y restricciones por vivienda.",
        agent=fin_agente, context=[t1],
    )
    t4 = Task(
        description="Consulta T1, T2 y T3. Invoca ranking_verificado sin argumentos para "
                    "contrastar cifras y excluir opciones inviables. Produce una "
                    "recomendación en español con ranking viable, explicación, "
                    "trade-offs y advertencia de que los datos son simulados. "
                    "No contradigas el ranking del motor verificado.",
        expected_output="Reporte Markdown con ranking, mensualidades, una sección explícita de exclusiones, "
                        "justificación y límites del prototipo.",
        agent=coord_agente, context=[t1, t2, t3],
    )

    def log_tarea(resultado_tarea) -> None:
        """Log de entregables entre agentes, SIN registrar pensamientos privados."""
        salida.mkdir(parents=True, exist_ok=True)
        with (salida / "log_crew_entregables.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n=== TAREA COMPLETADA ===\n{str(resultado_tarea)[:2400]}\n")

    return Crew(
        agents=[perfil_agente, zona_agente, fin_agente, coord_agente],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True,
        memory=False,
        planning=False,
        task_callback=log_tarea,
    )


def ejecutar_demo_con_evidencia(perfil: dict, viviendas: list[dict], salida: Path) -> dict:
    resultado, log = simulacion(perfil, viviendas)
    guardar_evidencia_base(salida, perfil, resultado, "demo")
    (salida / "log_demo.txt").write_text(log, encoding="utf-8")
    interacciones = construir_interacciones_agentes(perfil, viviendas, modo="demo")
    guardar_json(salida / "interaccion_agentes.json", interacciones)
    return {"ranking": resultado, "explicacion": log, "salida": salida}


def _posicion_mencion(texto: str, vivienda: dict) -> int:
    """Localiza una vivienda por ID o nombre para tolerar estilos distintos del LLM."""
    candidatos = (vivienda["id"].lower(), vivienda["nombre"].lower())
    posiciones = [texto.find(valor) for valor in candidatos if texto.find(valor) >= 0]
    return min(posiciones) if posiciones else -1


def evaluar_informe_crew(texto: str, ranking: dict) -> str:
    """Clasifica el informe como aceptado, incompleto reparable o contradictorio."""
    texto_normalizado = texto.lower()
    viables = sorted(
        (v for v in ranking["ranking"] if v["viable"]),
        key=lambda v: v["posicion"],
    )
    posiciones = [_posicion_mencion(texto_normalizado, vivienda) for vivienda in viables]
    if any(pos < 0 for pos in posiciones) or posiciones != sorted(posiciones):
        return "contradictorio"
    excluidas = [v for v in ranking["ranking"] if not v["viable"]]
    exclusiones_completas = all(
        _posicion_mencion(texto_normalizado, vivienda) >= 0
        and all(
            motivo.lower() in texto_normalizado
            for motivo in vivienda["motivos_exclusion"]
        )
        for vivienda in excluidas
    )
    return "aceptado" if exclusiones_completas else "incompleto_reparable"


def informe_crew_consistente(texto: str, ranking: dict) -> bool:
    """Compatibilidad booleana para consumidores y pruebas existentes."""
    return evaluar_informe_crew(texto, ranking) == "aceptado"


def completar_exclusiones_informe(texto: str, ranking: dict) -> str:
    """Añade datos deterministas omitidos sin descartar el informe útil."""
    lineas = [texto.rstrip(), "", "## Opciones excluidas verificadas", ""]
    for vivienda in ranking["ranking"]:
        if not vivienda["viable"]:
            lineas.append(
                f"- **{vivienda['id']} - {vivienda['nombre']}**: "
                f"{', '.join(vivienda['motivos_exclusion'])}; mensualidad "
                f"${vivienda['mensualidad_mxn']:,.2f} frente a un máximo de "
                f"${vivienda['pago_maximo_mxn']:,.2f}."
            )
    lineas.extend([
        "",
        "_Esta sección fue completada automáticamente con el motor determinista._",
    ])
    return "\n".join(lineas) + "\n"


def generar_informe_determinista(ranking: dict) -> str:
    """Reporte seguro usado cuando el texto local no supera la verificación."""
    lineas = [
        "# Recomendación validada por HabiMAS",
        "",
        "Los cuatro agentes sí se ejecutaron y su informe bruto se conservó como evidencia. "
        "La presentación fue reemplazada porque el texto del coordinador contradijo el "
        "ranking verificado; los cálculos deterministas no se modificaron.",
        "",
        "## Ranking viable",
        "",
    ]
    for vivienda in ranking["ranking"]:
        if vivienda["viable"]:
            lineas.append(
                f"{vivienda['posicion']}. **{vivienda['id']} - {vivienda['nombre']}**: "
                f"puntaje {vivienda['puntaje_final']}; mensualidad "
                f"${vivienda['mensualidad_mxn']:,.2f}."
            )
    lineas.extend(["", "## Opciones excluidas", ""])
    for vivienda in ranking["ranking"]:
        if not vivienda["viable"]:
            lineas.append(
                f"- **{vivienda['id']} - {vivienda['nombre']}**: "
                f"{', '.join(vivienda['motivos_exclusion'])}."
            )
    lineas.extend(["", f"> {ranking['advertencia']}"])
    return "\n".join(lineas) + "\n"


def ejecutar_crew_local(
    perfil: dict,
    viviendas: list[dict],
    salida: Path,
    modelo: str = MODELO_OLLAMA_DEFAULT,
    base_url: str = OLLAMA_BASE_URL_DEFAULT,
) -> dict:
    """Ejecuta agentes locales y conserva un ranking determinista autoritativo."""
    comprobar_ollama(modelo, base_url)
    ranking = construir_ranking(perfil, viviendas)
    guardar_evidencia_base(salida, perfil, ranking, "crew_local", modelo)
    try:
        respuesta = crear_crew(perfil, viviendas, salida, modelo, base_url).kickoff(inputs={})
        explicacion_bruta = str(respuesta)
        entregables = [
            str(getattr(tarea, "raw", tarea))
            for tarea in getattr(respuesta, "tasks_output", [])
        ]
        interacciones = construir_interacciones_agentes(
            perfil, viviendas, entregables=entregables, modo="crew_local"
        )
        guardar_json(salida / "interaccion_agentes.json", interacciones)
        guardar_json(salida / "ejecucion_agentes.json", {
            "framework": "CrewAI",
            "modelo": modelo,
            "agentes_configurados": 4,
            "tareas_completadas": len(entregables),
            "ejecucion_real": True,
            "herramientas": [
                "normalizar_perfil_usuario",
                "puntuar_alternativas_de_zona",
                "calcular_escenarios_hipotecarios",
                "ranking_verificado",
            ],
        })
        (salida / "informe_crew_bruto.md").write_text(explicacion_bruta, encoding="utf-8")
        estado_informe = evaluar_informe_crew(explicacion_bruta, ranking)
        if estado_informe == "aceptado":
            explicacion = explicacion_bruta
            accion = "aceptado"
        elif estado_informe == "incompleto_reparable":
            explicacion = completar_exclusiones_informe(explicacion_bruta, ranking)
            accion = "completado_con_exclusiones_deterministas"
        else:
            explicacion = generar_informe_determinista(ranking)
            accion = "sustituido_por_informe_determinista"
        (salida / "informe_crew.md").write_text(explicacion, encoding="utf-8")
        guardar_json(salida / "validacion_informe.json", {
            "estado": estado_informe,
            "consistente_con_ranking": estado_informe != "contradictorio",
            "accion": accion,
        })
        return {
            "ranking": ranking,
            "explicacion": explicacion,
            "salida": salida,
            "informe_crew_consistente": estado_informe != "contradictorio",
            "estado_informe": estado_informe,
        }
    except Exception as exc:
        (salida / "error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="HabiMAS: demo auditable o CrewAI real")
    parser.add_argument("--modo", choices=["demo", "crew"], default="demo")
    parser.add_argument("--perfil", type=Path, default=BASE / "perfil_usuario.json")
    parser.add_argument("--viviendas", type=Path, default=BASE / "viviendas_simuladas.json")
    parser.add_argument("--salida", type=Path, default=BASE / "salidas")
    parser.add_argument(
        "--modelo", choices=MODELOS_OLLAMA,
        default=os.getenv("CREW_MODEL", MODELO_OLLAMA_DEFAULT),
    )
    parser.add_argument(
        "--ollama-url", default=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT),
    )
    args = parser.parse_args()
    perfil = leer_json(args.perfil)
    viviendas = leer_json(args.viviendas)
    args.salida.mkdir(parents=True, exist_ok=True)

    if args.modo == "demo":
        resultado, log = simulacion(perfil, viviendas)
        guardar_json(args.salida / "ranking_demo.json", resultado)
        (args.salida / "log_demo.txt").write_text(log, encoding="utf-8")
        print(log)
        print(f"Salida JSON: {args.salida / 'ranking_demo.json'}")
        return

    salida_ejecucion = crear_directorio_ejecucion("crew_local", args.modelo)
    try:
        resultado = ejecutar_crew_local(
            perfil, viviendas, salida_ejecucion, args.modelo, args.ollama_url
        )
    except (ConnectionError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print("\n=== INFORME DEL CREW LOCAL ===\n", resultado["explicacion"])
    print(f"Evidencia y ranking validado: {salida_ejecucion}")


if __name__ == "__main__":
    main()
