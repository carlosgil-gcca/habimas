"""Prueba de humo de la interfaz sin ejecutar agentes ni red."""

from streamlit.testing.v1 import AppTest


def test_app_carga_en_modo_demo():
    app = AppTest.from_file("app.py", default_timeout=15).run()
    assert not app.exception
    assert any("HabiMAS" in item.value for item in app.markdown)
    assert [tab.label for tab in app.tabs] == [
        "⚙️ Configuración", "📊 Recomendación", "🤖 Interacción de agentes",
    ]
    assert app.radio[0].value == "Demo determinista"
    assert app.button[0].label == "Generar recomendación"
    assert sum(control.value for control in app.slider[:4]) == 100
    app.slider[0].set_value(60).run()
    assert [control.value for control in app.slider[:4]] == [60, 15, 15, 10]
    assert sum(control.value for control in app.slider[:4]) == 100
    app.button[0].click().run()
    assert not app.exception
    assert {item.label for item in app.metric} >= {
        "Efectivo para compra", "Pago hipotecario máximo", "Habitaciones mínimas",
        "Modo", "Etapas", "Evidencia",
    }
    assert "Ejecución terminada" in app.success[0].value
    assert any("Esta ejecución no utilizó un LLM" in item.value for item in app.warning)
    etiquetas = [item.label for item in app.expander]
    assert any(etiqueta.startswith("T1 ·") for etiqueta in etiquetas)
    assert any(etiqueta.startswith("T4 ·") for etiqueta in etiquetas)
