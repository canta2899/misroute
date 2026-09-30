from functools import lru_cache
from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=None)
def _template(name: str) -> Template:
    path = TEMPLATE_DIR / f"{name}.md"
    # Templates end with a newline as text files should; prompts do not.
    return Template(path.read_text(encoding="utf-8").rstrip("\n"))


def render(name: str, **values: object) -> str:
    return _template(name).substitute(values)
