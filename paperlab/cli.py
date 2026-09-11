import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .core import Costs, digest, load_ticks
from .news import FinBERT, News


def main():
    parser = argparse.ArgumentParser(description="Fly + compact PPO paper lab. No live orders.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=1200)
    p = commands.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--product", default="BTC-USD", choices=["BTC-USD", "ETH-USD"])
    p.add_argument("--days", type=int, default=14)
    p = commands.add_parser("news")
    p.add_argument("--db", required=True)
    p.add_argument("--feed", action="append")
    p.add_argument("--finbert-revision")
    for command in ("train", "fly-replay", "export"):
        p = commands.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--news-db")
        p.add_argument("--no-news", action="store_true")
        if command == "train":
            p.add_argument("--steps", type=int, default=8192)
            p.add_argument("--seed", type=int, default=7)
        if command == "fly-replay":
            p.add_argument("--fly-data", required=True)
            p.add_argument("--steps", type=int, default=8)
            p.add_argument("--start", type=int, default=63)
            p.add_argument("--frozen", action="store_true")
            p.add_argument("--checkpoint")
        if command == "export":
            p.add_argument("--model", required=True)
    p = commands.add_parser("fly-prepare")
    p.add_argument("--fly-data", required=True)
    p = commands.add_parser("tick")
    p.add_argument("--root", required=True)
    p.add_argument("--product", default="BTC-USD", choices=["BTC-USD", "ETH-USD"])
    p.add_argument("--fly", action="store_true")
    p.add_argument("--train-daily", action="store_true")
    p.add_argument("--model")
    args = parser.parse_args()
    if args.command in ("synthetic", "fetch"):
        from .data import synthetic, historical
        result = synthetic(args.out, args.count) if args.command == "synthetic" else historical(args.out, args.product, args.days)
    elif args.command == "news":
        from .news import DEFAULT_FEEDS
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        news = News(args.db)
        result = news.ingest(args.feed or DEFAULT_FEEDS, FinBERT(args.finbert_revision) if args.finbert_revision else None)
        news.db.close()
    elif args.command == "fly-prepare":
        from .fly import prepare
        result = prepare(args.fly_data)
    elif args.command == "tick":
        from .runtime import cycle
        result = cycle(args.root, args.product, args.fly, train_daily=args.train_daily, model=args.model)
    else:
        ticks = load_ticks(args.data)
        news = News(args.news_db or ":memory:", enabled=not args.no_news)
        try:
            if args.command == "train":
                from .compact import train
                result = train(ticks, news, args.out, steps=args.steps, seed=args.seed, dataset_sha=digest(args.data))
                result = {k: result[k] for k in ("parameters", "steps", "train_seconds", "model_sha256", "source")}
            elif args.command == "fly-replay":
                from .fly import replay
                result = replay(ticks, news, args.fly_data, args.out, args.steps, not args.frozen, args.checkpoint, start=args.start)
            else:
                from .export import export
                result = export(args.model, ticks, news, args.out)
        finally:
            news.db.close()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
