# HabiMAS | Prototipo académico multiagente

HabiMAS recomienda viviendas simuladas con cuatro agentes CrewAI y modelos locales en Ollama. El ranking y los cálculos financieros permanecen en un motor Python determinista; el modelo local coordina herramientas y explica los resultados, pero no decide las cifras.

## Alcance y modos

- **Demo determinista:** no usa CrewAI ni LLM; funciona sin Ollama.
- **Agentes locales:** ejecuta cuatro agentes CrewAI con `llama3.1:8b` o `llama3.2:latest`.
- **Streamlit:** permite editar el perfil, ejecutar ambos modos y comparar resultados.

Los precios, viviendas, puntuaciones territoriales, tasa y costos son **simulados**. El resultado no constituye una oferta hipotecaria ni una evaluación real de seguridad, servicios o movilidad.

## Requisitos externos

`requirements.txt` instala todas las dependencias Python, incluidos CrewAI y su cliente OpenAI-compatible. Python, Ollama y los archivos de los modelos no son paquetes Python y deben instalarse por separado.

1. Python 3.11.
2. [Ollama](https://ollama.com/download), sólo para agentes locales.
3. Modelos locales:

```bash
ollama pull llama3.1:8b
ollama pull llama3.2:latest
```

## Instalación

En macOS con Homebrew:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Interfaz Streamlit

Inicie Ollama si utilizará agentes locales:

```bash
ollama serve
```

En otra terminal, con el entorno virtual activo:

```bash
python -m streamlit run app.py
```

La interfaz organiza el flujo en tres pestañas: Configuración, Recomendación e Interacción de agentes. Incluye formulario financiero, prioridades territoriales vinculadas que siempre suman 100%, selector de modelo, ranking, gráficas, exclusiones y evidencia independiente por ejecución. La tercera pestaña explica el flujo T1–T4 y muestra los entregables observables de la última ejecución.

## Terminal

```bash
# Demo sin modelo
python habimas.py --modo demo

# CrewAI local recomendado
python habimas.py --modo crew --modelo llama3.1:8b

# Modelo local más rápido
python habimas.py --modo crew --modelo llama3.2:latest

# Servidor Ollama personalizado
python habimas.py --modo crew --modelo llama3.1:8b --ollama-url http://localhost:11434/v1
```

También se pueden definir `CREW_MODEL` y `OLLAMA_BASE_URL` siguiendo `.env.example`. El programa no carga automáticamente `.env`; las variables deben exportarse en la terminal.

## Arquitectura

1. **Agente de perfil:** valida requisitos y restricciones.
2. **Agente territorial:** explica las puntuaciones sintéticas de zona.
3. **Agente financiero:** consulta pagos y viabilidad calculados en Python.
4. **Coordinador:** redacta la recomendación y la contrasta con el ranking validado.

El modo local utiliza la API compatible con OpenAI de Ollama en `http://localhost:11434/v1`. No necesita una clave real ni recurre silenciosamente a un proveedor externo.

## Evidencia

Cada corrida nueva genera una carpeta única bajo `salidas/ejecuciones/` con el perfil, configuración, ranking validado e `interaccion_agentes.json`. Este último documenta para T1–T4 la entrada, herramienta, operaciones deterministas, salida verificable y entregable del agente. El modo local agrega el informe presentado, la salida bruta del Crew, el resultado de su validación y el log de entregables. Si el Crew falla después de iniciar, también se guarda `error.txt`.

El ranking validado es siempre la salida autoritativa. Si el texto local ordena correctamente las opciones pero omite exclusiones, la aplicación conserva el informe y completa sólo esa sección con el motor determinista. El informe completo se sustituye únicamente cuando contradice el ranking.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas cubren el motor original, modelos permitidos, errores de Ollama, persistencia sin sobrescritura y carga básica de Streamlit.

## Google Colab

El notebook conserva la demo determinista y el flujo original de clase. Ollama local no se ejecuta dentro de Colab: el modelo corre en la computadora del usuario y Streamlit se utiliza localmente.

## Límites metodológicos

- No se consultan precios, créditos, seguridad o servicios reales.
- La tasa, costos iniciales y límite de pago son hipótesis configurables.
- La mensualidad no incluye automáticamente seguros, CAT, predial o mantenimiento.
- Un modelo local pequeño puede ser más lento o menos consistente que uno remoto.
- Ninguna explicación sustituye el ranking determinista ni una evaluación profesional.
