"""Pruebas unitarias del motor (no requieren modelos de lenguaje)."""
import json
from pathlib import Path
import pytest
from habimas import (
    MODELOS_OLLAMA,
    comprobar_ollama,
    completar_exclusiones_informe,
    consultar_modelos_ollama,
    construir_interacciones_agentes,
    crear_directorio_ejecucion,
    construir_ranking,
    evaluar_informe_crew,
    generar_informe_determinista,
    informe_crew_consistente,
    mensualidad_francesa,
    validar_modelo_local,
    validar_perfil,
)

BASE = Path(__file__).resolve().parent
P = json.loads((BASE / 'perfil_usuario.json').read_text(encoding='utf8'))
V = json.loads((BASE / 'viviendas_simuladas.json').read_text(encoding='utf8'))

def test_perfil_capacidad():
    p = validar_perfil(P)
    assert p['pago_hipoteca_maximo_mxn'] == 19250
    assert p['efectivo_para_compra_mxn'] == 700000

def test_cuota_cero_y_tasa_cero():
    assert mensualidad_francesa(0, 0.10, 20) == 0
    assert mensualidad_francesa(240000, 0, 20) == 1000

def test_restriccion_dura_excluye_mejor_zona():
    r = construir_ranking(P, V)['ranking']
    assert len(r) == 10
    excluidas = {x['id']: x for x in r if not x['viable']}
    assert set(excluidas) == {'VIV-D', 'VIV-H', 'VIV-I'}
    assert excluidas['VIV-D']['puntaje_final'] is None
    assert 'Mensualidad supera capacidad de pago' in excluidas['VIV-D']['motivos_exclusion']
    assert 'Habitaciones insuficientes' in excluidas['VIV-H']['motivos_exclusion']
    assert r[0]['viable'] is True

def test_catalogo_simulado_es_diverso_y_tiene_ids_unicos():
    assert len(V) == 10
    assert len({v['id'] for v in V}) == len(V)
    assert min(v['precio_mxn'] for v in V) < 1800000
    assert max(v['precio_mxn'] for v in V) > 2900000
    assert {v['habitaciones'] for v in V} >= {1, 2, 3, 4}

def test_habitaciones_obligatorias():
    p = dict(P)
    p['habitaciones_minimas'] = 3
    r = construir_ranking(p, V)['ranking']
    assert all(x['habitaciones'] >= 3 for x in r if x['viable'])

def test_pesos_invalidos():
    p = dict(P)
    p['pesos_zona'] = dict(P['pesos_zona'], transporte=.5)
    with pytest.raises(ValueError):
        validar_perfil(p)


@pytest.mark.parametrize('modelo', MODELOS_OLLAMA)
def test_modelos_locales_admitidos(modelo):
    assert validar_modelo_local(modelo) == modelo


def test_modelo_local_no_admitido():
    with pytest.raises(ValueError, match='no admitido'):
        validar_modelo_local('modelo-inventado')


def test_error_controlado_si_ollama_no_responde(monkeypatch):
    def falla(*args, **kwargs):
        raise OSError('sin servidor')
    monkeypatch.setattr('urllib.request.urlopen', falla)
    with pytest.raises(ConnectionError, match='Inicia Ollama'):
        consultar_modelos_ollama()


def test_error_controlado_si_falta_modelo(monkeypatch):
    monkeypatch.setattr('habimas.consultar_modelos_ollama', lambda base_url: {'llama3.2:latest'})
    with pytest.raises(RuntimeError, match='ollama pull llama3.1:8b'):
        comprobar_ollama('llama3.1:8b')


def test_directorios_de_ejecucion_no_se_sobrescriben(tmp_path):
    a = crear_directorio_ejecucion('demo', raiz=tmp_path)
    b = crear_directorio_ejecucion('demo', raiz=tmp_path)
    assert a.is_dir() and b.is_dir()
    assert a != b


def test_informe_contradictorio_se_detecta_y_se_reemplaza():
    ranking = construir_ranking(P, V)
    texto_incorrecto = '1. VIV-D 2. VIV-C 3. VIV-B 4. VIV-A. Ninguna excluida.'
    assert informe_crew_consistente(texto_incorrecto, ranking) is False
    seguro = generar_informe_determinista(ranking)
    assert seguro.index('VIV-B') < seguro.index('VIV-C') < seguro.index('VIV-A')
    assert 'VIV-D' in seguro and 'Mensualidad supera capacidad de pago' in seguro


def test_informe_correcto_sin_exclusiones_se_completa_sin_descartarlo():
    ranking = construir_ranking(P, V)
    parcial = ''.join(
        f"{v['posicion']}. {v['nombre']}\n"
        for v in ranking['ranking'] if v['viable']
    )
    assert evaluar_informe_crew(parcial, ranking) == 'incompleto_reparable'
    completo = completar_exclusiones_informe(parcial, ranking)
    assert completo.startswith(parcial.rstrip())
    assert 'VIV-D' in completo
    assert 'VIV-H' in completo
    assert 'VIV-I' in completo
    assert 'Mensualidad supera capacidad de pago' in completo


def test_traza_estructurada_describe_cuatro_agentes():
    interacciones = construir_interacciones_agentes(P, V, modo='demo')
    assert [item['tarea'] for item in interacciones] == ['T1', 'T2', 'T3', 'T4']
    assert interacciones[0]['herramienta'] == 'normalizar_perfil_usuario'
    assert interacciones[2]['salida_herramienta'][3]['viable'] is False
    assert interacciones[3]['salida_herramienta']['ranking'][0]['id'] == 'VIV-B'
    assert all(item['tipo_entregable'] == 'simulado' for item in interacciones)
