from debate.aggregation import majority_vote


def test_majority_wins():
    assert majority_vote(["3", "5", "3"], "gsm8k") == ("3", 2)


def test_equivalent_answers_are_grouped():
    answer, votes = majority_vote([r"\frac{1}{2}", "0.25", "0.5"], "math500")
    assert answer == r"\frac{1}{2}"
    assert votes == 2


def test_tie_goes_to_first_answer():
    assert majority_vote(["7", "8"], "gsm8k") == ("7", 1)


def test_missing_answers_are_ignored():
    assert majority_vote([None, "4", None], "gsm8k") == ("4", 1)
    assert majority_vote([None, None], "gsm8k") == (None, 0)
