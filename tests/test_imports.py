def test_import_e88_model():
    from ndm.models.e88_fused import E88FusedLM

    assert E88FusedLM.__name__ == "E88FusedLM"


def test_import_eval_builder():
    import scripts.build_reasoning_eval_panel as panel

    assert "reclor" in panel.TASKS
