from pathlib import Path

from debate.experiment import ExperimentConfig
from debate.serving import ServerSpec, load_stage

ROOT = Path(__file__).parents[1]


def test_server_command_uses_half_precision_for_t4():
    command = ServerSpec(
        model="Qwen/Qwen2.5-1.5B-Instruct", port=8001, gpu=0, memory=0.85
    ).command()
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert command[command.index("--dtype") + 1] == "half"
    assert command[command.index("--port") + 1] == "8001"


def test_all_stages_are_consistent():
    for name in ("gsm8k", "math500", "hetero"):
        stage = load_stage(name, ROOT / "configs" / "stages.yaml")
        served = set(stage.endpoints)
        for path in stage.experiments:
            config = ExperimentConfig.from_yaml(ROOT / path)
            # chaque modèle utilisé par une expérience est servi par l'étape
            assert set(config.agent_models()) <= served, (name, path)
            # le tour 1 réutilisé provient d'une expérience lancée avant dans la même étape
            if config.initial_answers:
                source = config.initial_answers.name.removesuffix(".jsonl")
                earlier = [p.stem for p in stage.experiments[: stage.experiments.index(path)]]
                assert source in earlier, (name, path)


def test_gpu_memory_is_not_overcommitted():
    for name in ("gsm8k", "math500", "hetero"):
        stage = load_stage(name, ROOT / "configs" / "stages.yaml")
        for gpu in {s.gpu for s in stage.servers}:
            assert sum(s.memory for s in stage.servers if s.gpu == gpu) <= 0.9
        assert len({s.port for s in stage.servers}) == len(stage.servers)


def test_debate_prompts_fit_in_server_context():
    """Pire cas : question + propre réponse + réponses des autres + nouvelle génération."""
    for name in ("gsm8k", "math500", "hetero"):
        stage = load_stage(name, ROOT / "configs" / "stages.yaml")
        context = {s.model: s.max_model_len for s in stage.servers}
        for path in stage.experiments:
            config = ExperimentConfig.from_yaml(ROOT / path)
            agents = config.n_agents if config.method == "debate" else 1
            worst = 1000 + agents * config.max_tokens + config.max_tokens
            for model in set(config.agent_models()):
                assert worst <= context[model], (name, path, model)
