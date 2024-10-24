import logging

import pyfiglet
from rich.console import Console
from rich.text import Text


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


logger = get_logger("ITSPER")


def display_ascii_text_with_circle() -> None:
    console = Console()

    itsp_ascii_art = pyfiglet.figlet_format("ITSPER", font="slant").splitlines()

    circle = [
        Text("      ◯ ◯ ◯ ◯ ◯      ", style="bold red"),  # Top of the circle (red)
        Text("    ◯ ", style="bold red") + Text("◯ ◯ ", style="bold red") + Text("◯ ◯ ◯ ◯  ", style="bold green"),
        Text("  ◯ ◯ ", style="bold red")
        + Text("◯ ◯ ", style="bold yellow")
        + Text("◯ ◯ ◯ ", style="bold green")
        + Text("◯ ◯ ", style="bold red"),
        Text("  ◯ ◯ ", style="bold red")
        + Text("◯ ◯ ", style="bold yellow")
        + Text("◯ ◯ ", style="bold green")
        + Text("◯ ◯ ◯ ", style="bold red"),
        Text("    ◯ ", style="bold red") + Text("◯ ◯ ◯ ◯ ◯ ", style="bold green") + Text("◯ ", style="bold red"),
        Text("      ◯ ◯ ◯ ◯ ◯      ", style="bold red"),  # Bottom of the circle (red)
    ]

    combined_output = []
    for i in range(len(itsp_ascii_art)):
        if i < len(circle):
            combined_output.append((itsp_ascii_art[i], circle[i]))
        else:
            combined_output.append((itsp_ascii_art[i], None))

    for text, circle_text in combined_output:
        if circle_text:
            console.print(text + "   ", end="")
            console.print(circle_text)
        else:
            console.print(text)

    print_rich_statement(console)


def print_rich_statement(console: Console) -> None:
    statement = Text("ITSPER - ", style="bold blue")
    statement.append("Exploring tumor microenvironments through ", style="italic green")
    statement.append("ITSP quantification.", style="bold magenta")
    console.print(statement)


def display_launch_graphic() -> None:
    display_ascii_text_with_circle()
    logger.info("ITSPER is not intended for clinical use")
