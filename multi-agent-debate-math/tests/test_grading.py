import pytest

from debate.grading import equivalent, extract_answer, extract_boxed, is_correct, to_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"So the answer is \boxed{42}.", "42"),
        (r"\boxed{1} then finally \boxed{\frac{3}{2}}", r"\frac{3}{2}"),  # dernier boxed
        (r"\boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)"),
        (r"\fbox{7}", "7"),
        (r"The answer is \boxed{\frac{1}{", None),  # tronqué
        ("No box here: 12", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


def test_gsm8k_falls_back_to_last_number():
    assert extract_answer("She has 3 apples and buys 1,200 more.", "gsm8k") == "1200"
    assert extract_answer("No numbers.", "gsm8k") is None


def test_math_requires_boxed():
    assert extract_answer("The answer is 12.", "math500") is None


@pytest.mark.parametrize(
    ("gold", "prediction", "expected"),
    [
        ("18", r"\$18", True),
        ("18", "18.00", True),
        ("1234", "1,234", True),
        ("18", "18 dollars", True),
        ("50", r"50\%", True),
        ("0.5", r"\frac{1}{2}", True),
        ("18", "19", False),
        ("-3", "3", False),
        ("18", None, False),
    ],
)
def test_gsm8k_equivalence(gold, prediction, expected):
    assert equivalent(gold, prediction, "gsm8k") is expected


@pytest.mark.parametrize(
    ("gold", "prediction", "expected"),
    [
        (r"\frac{1}{2}", "0.5", True),
        (r"\frac{\pi}{2}", r"\dfrac{\pi}{2}", True),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \pi/2)", True),
        ("5", "x=5", True),
        (r"\text{(C)}", "C", True),
        (r"2\sqrt{3}", r"\sqrt{12}", True),
        (r"x^2+2x+1", r"(x+1)^2", True),
        (r"\{1,2\}", r"\{2,1\}", True),
        ("90", r"90^\circ", True),
        (r"\frac{1}{2}", r"\frac{1}{3}", False),
        ("3", "-3", False),
        ("[-2,7]", "(-2,7)", False),  # intervalle fermé vs ouvert
        ("p - q", "q + p", False),
    ],
)
def test_math_equivalence(gold, prediction, expected):
    assert equivalent(gold, prediction, "math500") is expected


def test_to_number():
    assert to_number(r"\$1,234.50") == 1234.5
    assert to_number(r"-\frac{3}{4}") == -0.75
    assert to_number("aucun") is None


def test_is_correct_end_to_end():
    response = "Step 1: 16 - 3 - 4 = 9. Step 2: 9 * 2 = 18.\nSo she makes \\boxed{\\$18}."
    assert is_correct(response, "18", "gsm8k")
    assert not is_correct(response, "20", "gsm8k")


def test_math_equivalence_outside_main_thread():
    """Régression : math-verify levait une erreur hors du thread principal (signal.alarm)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(equivalent, r"\frac{1}{2}", "0.5", "math500").result()
        assert not pool.submit(equivalent, r"\frac{1}{2}", "0.4", "math500").result()
