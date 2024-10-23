import logging
import pyfiglet
from rich.console import Console
from rich.text import Text
import time

def get_logger(name: str) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set logger to capture info level messages

    # Create a console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)
    return logger


def display_ascii_text_with_circle():
    """Display ITSPER text in a retro style with a balanced colored circle on the same line, adding yellow highlights."""
    console = Console()

    # Generate the "ITSPER" text in a retro style using pyfiglet
    itsp_ascii_art = pyfiglet.figlet_format("ITSPER", font="slant").splitlines()

    # Create a properly shaped colored circle using rich.Text with red, green, and yellow regions
    circle = [
        Text("      ◯ ◯ ◯ ◯ ◯      ", style="bold red"),  # Top of the circle (red)
        Text("    ◯ ", style="bold red") + Text("◯ ◯ ", style="bold red") + Text("◯ ◯ ◯ ◯  ", style="bold green"),
        Text("  ◯ ◯ ", style="bold red") + Text("◯ ◯ ", style="bold yellow") + Text("◯ ◯ ◯ ", style="bold green") + Text("◯ ◯ ", style="bold red"),
        Text("  ◯ ◯ ", style="bold red") + Text("◯ ◯ ", style="bold yellow") + Text("◯ ◯ ", style="bold green") + Text("◯ ◯ ◯ ", style="bold red"),
        Text("    ◯ ", style="bold red") + Text("◯ ◯ ◯ ◯ ◯ ", style="bold green") + Text("◯ ", style="bold red"),
        Text("      ◯ ◯ ◯ ◯ ◯      ", style="bold red"),  # Bottom of the circle (red)
    ]

    # Combine ASCII art and circle on the same line
    combined_output = []
    for i in range(len(itsp_ascii_art)):
        if i < len(circle):
            combined_output.append((itsp_ascii_art[i], circle[i]))
        else:
            combined_output.append((itsp_ascii_art[i], None))

    # Print the combined result
    for text, circle_text in combined_output:
        if circle_text:
            console.print(text + "   ", end="")
            console.print(circle_text)
        else:
            console.print(text)

    print_rich_statement(console)


def print_rich_statement(console):
    """Print the ITSPected statement below the graphics using rich styling."""
    statement = Text("ITSPER - ", style="bold blue")
    statement.append("Exploring tumor microenvironments through ", style="italic green")
    statement.append("ITSP quantification.", style="bold magenta")
    console.print(statement)


def display_launch_graphic():
    """Display ASCII text and the circle graphic in the terminal on the same line."""
    display_ascii_text_with_circle()

