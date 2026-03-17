"""Gradio web app for registering new microscopy experiments.

Creates a hash-named folder on the NAS with a metadata.json file,
then returns the path for the researcher to paste into the microscopy software.
Supports multi-day experiments where data is added across multiple sessions.

Experiment discovery is based on scanning UPLOAD_ROOT for subdirectories
containing a metadata.json — no central database file needed.
"""

import hashlib
import json
import os
import stat
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (from environment / .env file)
# ---------------------------------------------------------------------------

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/data/upload"))
# Display path: the real path on the host/NAS shown to the user.
# Falls back to UPLOAD_ROOT if not set (e.g. when running locally).
UPLOAD_ROOT_DISPLAY = Path(os.environ.get("UPLOAD_ROOT_DISPLAY", str(UPLOAD_ROOT)))
APP_TITLE = os.environ.get("APP_TITLE", "UMG Pharmacology – New Experiment")

# ---------------------------------------------------------------------------
# Dropdown options (users can still type custom values)
# ---------------------------------------------------------------------------

INSTRUMENTS = [
    "Yokogawa CV8000 spinning disk",
    "Yokogawa CQ1 spinning disk",
]
MODALITIES = [
    "Fluorescence",
    "Bright field",
    "Fluorescence + bright field",
]

# ---------------------------------------------------------------------------
# Experiment discovery (filesystem-based, no central DB)
# ---------------------------------------------------------------------------


def _scan_experiments() -> list[dict]:
    """Scan UPLOAD_ROOT for folders containing metadata.json, return sorted list."""
    experiments = []
    if not UPLOAD_ROOT.exists():
        return experiments
    for meta_file in UPLOAD_ROOT.glob("*/metadata.json"):
        try:
            metadata = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, ValueError, OSError):
            continue
        folder = meta_file.parent
        # Use folder mtime as creation time if not stored in metadata
        created = metadata.get("created", "")
        if not created:
            created = datetime.fromtimestamp(folder.stat().st_ctime).isoformat()
        experiments.append({
            "folder": str(folder),
            "experiment_id": folder.name,
            "created": created,
            "metadata": metadata,
        })
    # Sort by creation date, newest first
    experiments.sort(key=lambda e: e.get("created", ""), reverse=True)
    return experiments


def _display_path(internal_path: str) -> str:
    """Convert a container-internal path to the display path shown to the user."""
    try:
        rel = Path(internal_path).relative_to(UPLOAD_ROOT)
        return str(UPLOAD_ROOT_DISPLAY / rel)
    except ValueError:
        return internal_path


def _experiment_label(exp: dict) -> str:
    """Human-readable label for an experiment entry."""
    meta = exp["metadata"]
    multi = " [multi-day]" if meta.get("multi_day") else ""
    created = exp.get("created", "")[:16].replace("T", " ")
    display = _display_path(exp["folder"])
    parts = [
        created,
        meta.get("user", ""),
        meta.get("project", ""),
        meta.get("instrument", ""),
        meta.get("cell_type", ""),
        meta.get("well_type", ""),
        meta.get("treatment", ""),
    ]
    label = " | ".join(p for p in parts if p)
    return f"{label}{multi} — {display}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def make_experiment_id(metadata: dict) -> str:
    """Generate a deterministic hash from metadata (identical data = same ID)."""
    payload = json.dumps(metadata, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def create_experiment_folder(metadata: dict) -> Path:
    """Create a hash-named folder with metadata.json inside."""
    # Hash includes acquisition_date (already in metadata) but not the time
    exp_id = make_experiment_id(metadata)
    metadata["created"] = datetime.now().isoformat()  # stored for display only
    folder = UPLOAD_ROOT / exp_id
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return folder


def unlock_folder(folder: Path) -> None:
    """Make folder writable so microscopy software can save data."""
    folder.chmod(
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def lock_folder(folder: Path) -> None:
    """Make folder read-only after acquisition."""
    folder.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH
    )


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------


def register_experiment(user, project, instrument, modality,
                        cell_type, well_type, treatment, replicate,
                        prep_date, labels, notes, multi_day):
    """Validate inputs, create folder, return path."""
    missing = []
    if not user:
        missing.append("User")
    if not project:
        missing.append("Project")
    if not instrument:
        missing.append("Instrument")
    if not modality:
        missing.append("Modality")
    if not cell_type:
        missing.append("Cell type")
    if not well_type:
        missing.append("Well type")
    if missing:
        raise gr.Error(f"Missing required fields: {', '.join(missing)}")

    metadata = {
        "user": user,
        "project": project,
        "instrument": instrument,
        "modality": modality,
        "cell_type": cell_type,
        "well_type": well_type,
        "treatment": treatment or "",
        "replicate": int(replicate) if replicate else 1,
        "prep_date": str(prep_date) if prep_date else "",
        "labels": labels or "",
        "notes": notes or "",
        "multi_day": bool(multi_day),
        "acquisition_date": str(date.today()),
    }

    folder = create_experiment_folder(metadata)
    unlock_folder(folder)

    return _display_path(str(folder))


def load_experiment_list():
    """Return list of experiment labels for the dropdown, sorted by date (newest first)."""
    experiments = _scan_experiments()
    if not experiments:
        return gr.update(choices=["(no experiments found)"], value=None)
    labels = [_experiment_label(exp) for exp in experiments]
    return gr.update(choices=labels, value=None)


def continue_experiment(selection):
    """Look up the selected experiment and return its folder path."""
    if not selection or selection == "(no experiments found)":
        return "", ""
    # The display path is the last part after " — "
    display_path = selection.split(" — ", 1)[-1]

    # Map display path back to internal path
    try:
        rel = Path(display_path).relative_to(UPLOAD_ROOT_DISPLAY)
        folder = UPLOAD_ROOT / rel
    except ValueError:
        folder = Path(display_path)

    if not folder.exists():
        raise gr.Error(f"Folder no longer exists: {display_path}")

    # Ensure it's writable for the new acquisition session
    unlock_folder(folder)

    # Load and display metadata
    meta_file = folder / "metadata.json"
    meta_text = meta_file.read_text() if meta_file.exists() else "{}"

    return display_path, meta_text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Microscopy Experiment Manager") as app:
    gr.Markdown(
        "# Microscopy Experiment Manager\n"
        "Register new microscopy experiments and manage multi-day acquisitions. "
        "Fill in the metadata below to create a uniquely named folder on the NAS, "
        "then paste the folder path into your microscopy software."
    )

    with gr.Tabs():
        # ==== Tab 1: New Experiment ====
        with gr.TabItem("New Experiment"):
            with gr.Group():
                gr.Markdown("### Experiment")
                with gr.Row():
                    user = gr.Textbox(label="User")
                    project = gr.Textbox(label="Project / Grant")
                with gr.Row():
                    instrument = gr.Dropdown(
                        INSTRUMENTS, label="Instrument",
                        value=INSTRUMENTS[0], allow_custom_value=True)
                    modality = gr.Dropdown(
                        MODALITIES, label="Modality",
                        value=MODALITIES[0], allow_custom_value=True)

            with gr.Group():
                gr.Markdown("### Sample")
                with gr.Row():
                    cell_type = gr.Textbox(label="Cell type")
                    well_type = gr.Textbox(label="Well type")
                with gr.Row():
                    treatment = gr.Textbox(label="Treatment / Condition")
                    replicate = gr.Number(label="Replicate #", value=1, precision=0)
                    prep_date = gr.DateTime(label="Preparation date", type="string")

            with gr.Group():
                gr.Markdown("### Optional")
                with gr.Row():
                    labels = gr.Textbox(label="Staining / Labels")
                    notes = gr.Textbox(label="Notes", lines=2)
                multi_day = gr.Checkbox(
                    label="Multi-day measurement (data will be added across multiple sessions)",
                    value=False)

            btn = gr.Button("Create Experiment Folder", variant="primary")
            output = gr.Code(
                label="Folder path — copy this into the microscopy software",
                interactive=False, language=None,
            )

            btn.click(
                register_experiment,
                inputs=[user, project, instrument, modality,
                        cell_type, well_type, treatment, replicate,
                        prep_date, labels, notes, multi_day],
                outputs=output,
            )

        # ==== Tab 2: Continue Experiment ====
        with gr.TabItem("Continue Experiment") as continue_tab:
            gr.Markdown("### Resume a previous experiment")
            gr.Markdown("Select an experiment to retrieve its folder path "
                        "(e.g. for multi-day measurements).")

            refresh_btn = gr.Button("Refresh list", variant="secondary")
            exp_dropdown = gr.Dropdown(
                label="Select experiment", choices=[], interactive=True)
            continue_btn = gr.Button("Get folder path", variant="primary")

            cont_output = gr.Code(
                label="Folder path — copy this into the microscopy software",
                interactive=False, language=None,
            )
            cont_meta = gr.Code(
                label="Experiment metadata", language="json", interactive=False)

            refresh_btn.click(load_experiment_list, outputs=exp_dropdown)
            continue_btn.click(
                continue_experiment, inputs=exp_dropdown,
                outputs=[cont_output, cont_meta],
            )

            # Load list on tab select and on page load
            continue_tab.select(load_experiment_list, outputs=exp_dropdown)
            app.load(load_experiment_list, outputs=exp_dropdown)

if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
