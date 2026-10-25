"""Assigned stances: two agents argue opposite sides of a proposal on purpose, a third weighs it.

Two models given the same neutral question tend to give the same middling answer, and a
second opinion that agrees by default is worth little. In consensus mode one agent makes
the strongest honest case for the proposal and the other the strongest honest case against
it, an agent of a third model family assesses it impartially without seeing either, and the
synthesiser decides and says what would reverse the decision.
"""
from __future__ import annotations

FOR = "for"
AGAINST = "against"
NEUTRAL = "neutral"

_INSTRUCTIONS = {
    FOR: (
        "[YOUR STANCE: FOR]\n"
        "Another agent is arguing against; you argue FOR the proposal in the task (if the task "
        "offers options, for the first one). Make the strongest honest case: the concrete benefits, "
        "the conditions under which it is the right choice, and how its known risks are handled.\n"
        "Honesty comes before the stance: do not invent benefits, and if the proposal is a bad idea "
        "say so plainly instead of defending it.\n"
        "End with one line starting `What would change my mind:`."
    ),
    AGAINST: (
        "[YOUR STANCE: AGAINST]\n"
        "Another agent is arguing for; you argue AGAINST the proposal in the task (if the task "
        "offers options, against the first one). Make the strongest honest case: the concrete costs "
        "and failure modes, the conditions under which it is the wrong choice, and the best "
        "alternative.\n"
        "Honesty comes before the stance: do not invent problems, and if the proposal is clearly "
        "right say so plainly instead of attacking it.\n"
        "End with one line starting `What would change my mind:`."
    ),
    NEUTRAL: (
        "[YOUR STANCE: NEUTRAL]\n"
        "Two other agents are arguing for and against the proposal in the task; you do not see "
        "their answers. You are the impartial assessor: weigh the proposal on its merits, name the "
        "two or three factors that actually decide it, and say which way you lean and how strongly.\n"
        "Do not split the difference for the sake of balance: if one side is clearly right, say so.\n"
        "End with one line starting `What would change my mind:`."
    ),
}


def neutral_alias(taken: list[str]) -> str | None:
    """The alias of the impartial third agent ([consensus] neutral_alias; "" switches it off).

    None when it is switched off, is one of the two advocates, or cannot run.
    """
    from orchestra import config
    from orchestra.providers.fallback import ALIAS_TABLE

    alias = str(config.get("consensus", "neutral_alias", "oll-coder") or "")
    if not alias or alias in taken or alias not in ALIAS_TABLE or not ALIAS_TABLE[alias][0].is_available():
        return None
    return alias


def assign(aliases: list[str]) -> dict[str, str]:
    """Stance of each agent: the first argues for, the second against, a third is neutral."""
    stances = {aliases[0]: FOR, aliases[1]: AGAINST}
    if len(aliases) > 2:
        stances[aliases[2]] = NEUTRAL
    return stances


def with_stance(prompt: str, stance: str) -> str:
    """The task followed by the stance the agent is to take."""
    return f"{prompt}\n\n{_INSTRUCTIONS[stance]}"


def verifier_note() -> str:
    return ("The candidates were assigned stances (for / against, and possibly a neutral assessor) on "
            "purpose: their disagreement with each other is intended and is not an issue.")


def synthesis_note(stances: dict[str, str]) -> str:
    who = ", ".join(f"{alias} argued {stance}" for alias, stance in stances.items())
    return (
        "[ASSIGNED STANCES]\n"
        f"{who}. The advocates disagree by design, so agreement between them is strong evidence "
        "and disagreement is expected. A neutral assessor, if present, answered without seeing them: "
        "treat its lean as one independent vote, not as the verdict.\n"
        "Do not average the two. In `## Answer` make a decision and state what evidence would "
        "reverse it. In `## Dissent` give the strongest argument of the side you decided against, "
        "in its own words.\n"
    )
