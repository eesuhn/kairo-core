import justsdk
import re
import subprocess

from pathlib import Path
from config._constants import ROOT_PATH


def check_and_update_headings(file_path: Path) -> bool:
    """
    Check if headings need updating and update them if necessary.

    Returns:
        bool: True if file was modified, False otherwise
    """
    if file_path.stat().st_size == 0:
        justsdk.print_info(f"File {file_path} is empty, skipping...")
        return False

    notebook = justsdk.read_file(file_path)

    if "cells" not in notebook:
        justsdk.print_info(f"No cells found in {file_path}, skipping...")
        return False

    expected_count = 0
    needs_update = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            for line in source:
                if line.startswith("### "):
                    expected_count += 1
                    heading_text = line[4:].strip()
                    numbered_pattern = r"^(\d+)\.\s+"
                    match = re.match(numbered_pattern, heading_text)

                    if match:
                        current_number = int(match.group(1))
                        if current_number != expected_count:
                            needs_update = True
                            break
                    else:
                        needs_update = True
                        break

            if needs_update:
                break

    if not needs_update:
        justsdk.print_info(f"Skipping {file_path}")
        return False

    count = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            new_lines: list = []
            for line in source:
                if line.startswith("### "):
                    count += 1
                    heading_text = line[4:].strip()
                    numbered_pattern = r"^(\d+)\.\s+"
                    if re.match(numbered_pattern, heading_text):
                        heading_text = re.sub(numbered_pattern, "", heading_text)

                    line_ending = "\n" if line.endswith("\n") else ""
                    new_line = f"### {count}. {heading_text}{line_ending}"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)

            cell["source"] = new_lines

    justsdk.write_file(notebook, file_path, indent=1)
    justsdk.print_success(f"Updated headings in {file_path}")
    return True


def get_staged_notebooks() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )

        staged_notebooks = []
        for filename in result.stdout.strip().split("\n"):
            if (
                filename
                and filename.endswith(".ipynb")
                and filename.startswith("notebooks/")
            ):
                file_path = ROOT_PATH / filename
                if file_path.exists():
                    staged_notebooks.append(file_path)

        return staged_notebooks

    except subprocess.CalledProcessError as e:
        justsdk.print_error(f"Git command failed: {e}")
        return []


def update_number_headings() -> None:
    staged_notebooks = get_staged_notebooks()

    if not staged_notebooks:
        justsdk.print_info("No staged notebooks")
        return

    justsdk.print_info(f"Checking {len(staged_notebooks)} staged notebook(s)...")

    modified_files = []

    for notebook_path in staged_notebooks:
        if check_and_update_headings(notebook_path):
            modified_files.append(notebook_path)

    if modified_files:
        try:
            subprocess.run(
                ["git", "add"] + [str(f) for f in modified_files], check=True
            )
            justsdk.print_success(f"Re-staged {len(modified_files)} modified file(s)")
        except subprocess.CalledProcessError as e:
            justsdk.print_error(f"Failed to re-stage files: {e}")
            return

    return
