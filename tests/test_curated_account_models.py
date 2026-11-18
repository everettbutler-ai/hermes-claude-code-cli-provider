"""Curated connection catalog contract, independent of live credentials."""
import claude_code_server as server


def test_retired_sonnet_is_not_selectable():
    assert 'claude-sonnet-5' not in server.ADVERTISED_MODELS
    assert 'claude-sonnet-5-5' in server.ADVERTISED_MODELS


def test_pinned_fleet_models_are_advertised_without_rewriting():
    for model in ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-fable-5-1', 'haiku'):
        assert model in server.ADVERTISED_MODELS
        assert server.normalize_model(model) == model
