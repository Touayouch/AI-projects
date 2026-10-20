"""Ligne de commande du projet.

debate ping "Combien font 17 × 23 ?"   # vérifie la connexion au serveur LLM
"""

import argparse
import asyncio
import logging
from pathlib import Path

from debate.config import get_settings
from debate.data import load_problems, load_subset, make_subset
from debate.experiment import ExperimentConfig, prepare_output, run_experiment
from debate.grading import extract_answer, is_correct
from debate.llm import LLMClient
from debate.metrics import (
    compare_at_budget,
    debate_summary,
    format_comparison,
    format_debate,
    format_summary,
    summarize,
)
from debate.prompts import solve_prompt
from debate.records import read_records

SUBSETS_DIR = Path("configs/subsets")


def cmd_ping(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = LLMClient(
        settings.llm_base_url, settings.llm_api_key, timeout=settings.request_timeout
    )
    completion = asyncio.run(
        client.complete(
            [{"role": "user", "content": args.prompt}],
            model=args.model or settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
        )
    )
    print(completion.text)
    print(
        f"\n[{completion.model}] {completion.prompt_tokens} tokens en entrée, "
        f"{completion.completion_tokens} en sortie, {completion.latency_s:.1f} s"
    )


def cmd_subset(args: argparse.Namespace) -> None:
    problems = load_problems(args.dataset, get_settings().data_dir)
    ids = make_subset(problems, args.size, args.seed)
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBSETS_DIR / f"{args.name or f'{args.dataset}_{len(ids)}'}.txt"
    path.write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} problèmes sur {len(problems)} -> {path}")


def cmd_check_grader(args: argparse.Namespace) -> None:
    """Chaque solution de référence doit être jugée correcte par le vérificateur."""
    failures = 0
    for dataset in ("gsm8k", "math500"):
        problems = load_problems(dataset, get_settings().data_dir)
        bad = [p for p in problems if not is_correct(p.solution, p.answer, dataset)]
        failures += len(bad)
        print(f"{dataset} : {len(problems) - len(bad)}/{len(problems)} solutions reconnues")
        for p in bad[:10]:
            print(f"  {p.id} attendu={p.answer!r} extrait={extract_answer(p.solution, dataset)!r}")
    raise SystemExit(1 if failures else 0)


def cmd_solve(args: argparse.Namespace) -> None:
    settings = get_settings()
    problems = load_subset(args.dataset, settings.data_dir, args.subset)[: args.limit]
    client = LLMClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.request_timeout,
        settings.max_concurrency,
    )
    model = args.model or settings.default_model

    async def run():
        return await asyncio.gather(
            *(
                client.complete(
                    solve_prompt(p.question),
                    model,
                    settings.temperature,
                    settings.max_tokens,
                    settings.seed,
                )
                for p in problems
            )
        )

    completions = asyncio.run(run())
    correct = 0
    for problem, completion in zip(problems, completions, strict=True):
        answer = extract_answer(completion.text, args.dataset)
        ok = is_correct(completion.text, problem.answer, args.dataset)
        correct += ok
        print(
            f"{'✓' if ok else '✗'} {problem.id}  attendu={problem.answer!r}  extrait={answer!r}  "
            f"({completion.completion_tokens} tokens, {completion.finish_reason})"
        )
    print(f"\nExactitude : {correct}/{len(problems)}")


def cmd_run(args: argparse.Namespace) -> None:
    settings = get_settings()
    config = ExperimentConfig.from_yaml(args.config)
    problems = load_subset(config.dataset, settings.data_dir, config.subset)[: args.limit]
    out_path = prepare_output(config, settings.results_dir)
    client = LLMClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.request_timeout,
        settings.max_concurrency,
    )
    asyncio.run(run_experiment(client, config, problems, out_path))
    print(f"Résultats : {out_path}")


def make_client(endpoints: dict[str, str] | None = None) -> LLMClient:
    settings = get_settings()
    return LLMClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.request_timeout,
        settings.max_concurrency,
        endpoints=settings.llm_endpoints | (endpoints or {}),
    )


def cmd_stage(args: argparse.Namespace) -> None:
    """Lance les serveurs vLLM de l'étape, puis ses expériences dans l'ordre (reprenables)."""
    from debate.serving import load_stage, start_server, wait_until_ready

    settings = get_settings()
    stage = load_stage(args.stage)
    print(f"Étape {stage.name} : {stage.description}")
    processes = []
    try:
        if not args.no_serve:
            processes = [start_server(s, settings.results_dir / "logs") for s in stage.servers]
            for spec, process in zip(stage.servers, processes, strict=True):
                wait_until_ready(spec, process)
        client = make_client(stage.endpoints)
        for path in stage.experiments:
            config = ExperimentConfig.from_yaml(path)
            problems = load_subset(config.dataset, settings.data_dir, config.subset)
            out_path = prepare_output(config, settings.results_dir)
            asyncio.run(run_experiment(client, config, problems, out_path))
    finally:
        for process in processes:
            process.terminate()


def cmd_report(args: argparse.Namespace) -> None:
    records = read_records(args.results)
    if records[0]["method"] == "debate":
        print(format_debate(args.results.stem, debate_summary(records)))
        return
    n = min(len(r["calls"]) for r in records)
    ks = args.ks or [k for k in (1, 2, 3, 4, 5, 8, 12, 16) if k <= n]
    print(format_summary(args.results.stem, summarize(records, ks)))


def cmd_compare(args: argparse.Namespace) -> None:
    comparison = compare_at_budget(read_records(args.debate), read_records(args.sampling))
    print(format_debate(args.debate.stem, comparison["debate"]))
    print(format_comparison(comparison))


def main() -> None:
    parser = argparse.ArgumentParser(prog="debate")
    sub = parser.add_subparsers(required=True)

    ping = sub.add_parser("ping", help="Teste la connexion au serveur LLM")
    ping.add_argument("prompt", nargs="?", default="Combien font 17 × 23 ? Réponds en une ligne.")
    ping.add_argument("--model")
    ping.set_defaults(func=cmd_ping)

    subset = sub.add_parser("subset", help="Fige un sous-ensemble reproductible de problèmes")
    subset.add_argument("dataset", choices=["gsm8k", "math500"])
    subset.add_argument("--size", type=int, required=True)
    subset.add_argument("--seed", type=int, default=0)
    subset.add_argument("--name", help="Nom du fichier (sans extension)")
    subset.set_defaults(func=cmd_subset)

    sub.add_parser(
        "check-grader", help="Valide le vérificateur sur les solutions de référence"
    ).set_defaults(func=cmd_check_grader)

    solve = sub.add_parser("solve", help="Réponse unique sur quelques problèmes (essai)")
    solve.add_argument("dataset", choices=["gsm8k", "math500"])
    solve.add_argument("--subset", type=Path, required=True)
    solve.add_argument("--limit", type=int, default=5)
    solve.add_argument("--model")
    solve.set_defaults(func=cmd_solve)

    run = sub.add_parser("run", help="Lance ou reprend une expérience (fichier YAML)")
    run.add_argument("config", type=Path)
    run.add_argument("--limit", type=int, help="Nombre maximum de problèmes (essai)")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="Résumé d'une expérience")
    report.add_argument("results", type=Path)
    report.add_argument("--ks", type=int, nargs="+", help="Valeurs de k pour vote@k")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="Compare un débat au vote à budget égal")
    compare.add_argument("debate", type=Path)
    compare.add_argument("sampling", type=Path)
    compare.set_defaults(func=cmd_compare)

    stage = sub.add_parser("stage", help="Étape d'expériences sur GPU (configs/stages.yaml)")
    stage.add_argument("stage")
    stage.add_argument("--no-serve", action="store_true", help="Serveurs déjà lancés")
    stage.set_defaults(func=cmd_stage)

    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level)
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
