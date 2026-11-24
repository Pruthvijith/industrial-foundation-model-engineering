"""Synthetic lab exercises with deterministic, equipment-separated splits."""

import hashlib
import json
import random

UNITS = ("W", "Pa", "J", "insufficient")
TEMPLATES = {
    "train": (
        "A heater draws {a} A at {b} V. Select the unit of electrical power.",
        "A piston presses with {a} N on {b} square metres. Select the unit of pressure.",
        "A motor delivers {a} W for {b} seconds. Select the unit of energy.",
        "A label reads {a} and {b}. No quantity is identified. Which unit applies?",
    ),
    "test": (
        "Current is {a} amperes and voltage is {b} volts. In what SI unit is their product?",
        "Divide {a} newtons by an area of {b} square metres. Give the resulting SI unit.",
        "Integrate a constant {a} watts over {b} seconds. Which derived unit fits?",
        "The log gives two readings, {a} and {b}, with missing headings. Identify the measured unit.",
    ),
}
LETTERS = "ABCD"


def cases(split: str, groups: int, seed: int = 1729) -> list[dict]:
    if groups < 1 or split not in TEMPLATES:
        raise ValueError("Expected a positive group count and train or test split")
    rows = []
    for group in range(groups):
        asset = f"{split}-{group:03}"
        rng = random.Random(f"{seed}/{asset}")
        for kind, template in enumerate(TEMPLATES[split]):
            options = list(UNITS)
            rng.shuffle(options)
            question = template.format(a=rng.randint(2, 90), b=rng.randint(2, 80))
            choices = ", ".join(f"{letter}={unit}" for letter, unit in zip(LETTERS, options))
            prompt = f"Lab record {asset}. {question}\nOptions: {choices}\nLetter:"
            label = LETTERS[options.index(UNITS[kind])]
            rows.append(
                dict(
                    id=f"{asset}/{kind}",
                    group=asset,
                    kind=UNITS[kind],
                    prompt=prompt,
                    target=label,
                    rejected=rng.choice(LETTERS.replace(label, "")),
                )
            )
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    if not rows or any(len({r[k] for r in rows}) != len(rows) for k in ("id", "prompt")):
        raise ValueError("Empty or duplicate records")
    for r in rows:
        if (
            r["target"] not in tuple(LETTERS)
            or r["rejected"] not in tuple(LETTERS)
            or r["target"] == r["rejected"]
        ):
            raise ValueError("Invalid preference")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
