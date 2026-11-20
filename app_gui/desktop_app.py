from __future__ import annotations

import sys
import ssl
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QButtonGroup,
    QToolButton,
    QWidget,
)

import app as engine
from session_store import load_session, save_session


APP_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()
VIU_ORANGE = "#E65014"
VIU_DARK = "#222330"
INPUT_FIELD_KEYS = (
    "time_col", "flux_col", "error_col", "flux_is_magnitude",
    "time_unit", "barycentric_mode", "target_name", "target_ra_deg", "target_dec_deg",
    "xmin", "xmax", "ymin", "ymax", "fits_extension", "fits_time_col", "fits_flux_col", "fits_error_col",
)
WINDOW_INPUT_KEYS = (
    "time_col", "time_unit", "barycentric_mode", "target_name", "target_ra_deg", "target_dec_deg",
    "xmin", "xmax", "fits_extension", "fits_time_col",
)


def sampling_window_signature(filename: str, fields: dict[str, str]) -> tuple:
    keys = list(WINDOW_INPUT_KEYS)
    if fields.get("ymin") or fields.get("ymax"):
        keys.extend(("flux_col", "ymin", "ymax", "fits_flux_col"))
    return filename, tuple((key, str(fields.get(key, ""))) for key in keys)


class WorkbenchState(QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.filename = ""
        self.file_bytes = b""
        self.fields: dict[str, str] = {}
        self.analysis_input_fields: dict[str, str] | None = None
        self.sampling_window_cache: dict | None = None
        self.result: dict | None = None
        self.evidence: dict[str, str] = {}

    def notify(self) -> None:
        self.changed.emit()

    def snapshot(self) -> dict:
        result = self.result
        if result is not None:
            result = {key: value for key, value in result.items() if key != "plots"}
        return {
            "filename": self.filename,
            "file_bytes": self.file_bytes,
            "fields": dict(self.fields),
            "analysis_input_fields": dict(self.analysis_input_fields) if self.analysis_input_fields is not None else None,
            "result": result,
            "evidence": dict(self.evidence),
        }

    def restore(self, snapshot: dict) -> None:
        self.filename = str(snapshot.get("filename", ""))
        self.file_bytes = snapshot.get("file_bytes", b"")
        self.fields = dict(snapshot.get("fields", {}))
        self.result = snapshot.get("result")
        committed = snapshot.get("analysis_input_fields")
        if committed is None and self.result is not None and self.file_bytes:
            committed = {key: self.fields[key] for key in INPUT_FIELD_KEYS if key in self.fields}
        if committed is not None:
            default_subtracted = bool((self.result or {}).get("flux_baseline_subtracted", False))
            committed.setdefault("subtract_baseline", str(default_subtracted).lower())
        self.analysis_input_fields = dict(committed) if committed is not None else None
        self.sampling_window_cache = None
        if self.result is not None and self.analysis_input_fields is not None:
            series = self.result.get("series", {})
            frequency = series.get("frequency", [])
            window_power = series.get("window_power", [])
            if len(frequency) > 1 and len(window_power) == len(frequency):
                self.sampling_window_cache = {
                    "frequency": frequency,
                    "power": window_power,
                    "signature": sampling_window_signature(self.filename, self.analysis_input_fields),
                }
        self.evidence = dict(snapshot.get("evidence", {}))
        self.notify()


class WorkThread(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(self, function, *args, context=None, **kwargs) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.context = context

    def run(self) -> None:
        try:
            self.succeeded.emit(self.function(*self.args, **self.kwargs), self.context)
        except Exception as exc:  # Surface worker errors in the main window.
            self.failed.emit(str(exc))


def double_box(value: float, minimum: float = 0, maximum: float = 1e12, decimals: int = 6, step: float = 0.01) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def int_box(value: int, minimum: int, maximum: int, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class PlotNavigationBar(QWidget):
    """Compact pan/box-zoom controls for a pyqtgraph plot, similar to Matplotlib's toolbar."""

    def __init__(self, plot: pg.PlotWidget, parent=None) -> None:
        super().__init__(parent)
        self.plot = plot
        self.view = plot.getViewBox()
        self._home_range: tuple[tuple[float, float], tuple[float, float]] | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.pan_button = self._mode_button("Pan", "Drag to move around the plot")
        self.zoom_button = self._mode_button("Box", "Drag a rectangle to zoom into that region")
        self.mode_group.addButton(self.pan_button)
        self.mode_group.addButton(self.zoom_button)
        self.pan_button.setChecked(True)
        self.pan_button.toggled.connect(lambda checked: checked and self.view.setMouseMode(pg.ViewBox.PanMode))
        self.zoom_button.toggled.connect(lambda checked: checked and self.view.setMouseMode(pg.ViewBox.RectMode))
        layout.addWidget(self.pan_button)
        layout.addWidget(self.zoom_button)
        self.zoom_in_button = self._action_button("+", "Zoom in")
        self.zoom_out_button = self._action_button("−", "Zoom out")
        self.home_button = self._action_button("Home", "Return to the original plot extent")
        self.zoom_in_button.clicked.connect(lambda: self._scale_view(0.8))
        self.zoom_out_button.clicked.connect(lambda: self._scale_view(1.25))
        self.home_button.clicked.connect(self.reset_view)
        layout.addWidget(self.zoom_in_button)
        layout.addWidget(self.zoom_out_button)
        layout.addWidget(self.home_button)
        layout.addStretch(1)

    @staticmethod
    def _mode_button(label: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(label)
        button.setCheckable(True)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    @staticmethod
    def _action_button(label: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(label)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    def _scale_view(self, factor: float) -> None:
        x_range, y_range = self.view.viewRange()
        center_x = (x_range[0] + x_range[1]) / 2.0
        center_y = (y_range[0] + y_range[1]) / 2.0
        half_x = (x_range[1] - x_range[0]) * factor / 2.0
        half_y = (y_range[1] - y_range[0]) * factor / 2.0
        self.view.setRange(
            xRange=(center_x - half_x, center_x + half_x),
            yRange=(center_y - half_y, center_y + half_y),
            padding=0,
        )

    def set_home_from_data(self, x_values, y_values, fixed_x_range=None) -> None:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            self.clear_home_range()
            return
        x = x[valid]
        y = y[valid]

        def padded(values: np.ndarray) -> tuple[float, float]:
            low = float(np.min(values))
            high = float(np.max(values))
            span = high - low
            padding = 0.02 * span if span > 0 else max(abs(low) * 0.02, 0.5)
            return low - padding, high + padding

        x_range = tuple(fixed_x_range) if fixed_x_range is not None else padded(x)
        self._home_range = (x_range, padded(y))
        self.view.setRange(xRange=x_range, yRange=self._home_range[1], padding=0)

    def clear_home_range(self) -> None:
        self._home_range = None

    def reset_view(self) -> None:
        if self._home_range is None:
            self.plot.autoRange(padding=0.02)
            return
        self.view.setRange(xRange=self._home_range[0], yRange=self._home_range[1], padding=0)


def resolve_target_name(name: str) -> tuple[float, float, str]:
    from astropy.coordinates import SkyCoord

    target = name.strip()
    if not target:
        raise ValueError("Enter an object name before searching SIMBAD.")
    coordinate = SkyCoord.from_name(target).icrs
    return float(coordinate.ra.deg), float(coordinate.dec.deg), coordinate.to_string("hmsdms", precision=2)


def bibliography_urls(object_name: str) -> dict[str, str]:
    keywords = '"time series" OR period OR periods OR periodicity OR timing OR variability'
    ads_query = f'"{object_name}" AND ({keywords})'
    arxiv_query = f'all:"{object_name}" AND (all:"time series" OR all:period OR all:periodicity OR all:timing OR all:variability)'
    return {
        "ADS": "https://ui.adsabs.harvard.edu/search/q=" + quote(ads_query),
        "arXiv": "https://arxiv.org/search/astro-ph?query=" + quote(object_name) + "&searchtype=all",
        "api": "https://export.arxiv.org/api/query?" + urlencode({
            "search_query": arxiv_query, "start": 0, "max_results": 20,
            "sortBy": "relevance", "sortOrder": "descending",
        }),
    }


def bibliography_search(object_name: str) -> list[dict[str, str]]:
    urls = bibliography_urls(object_name)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    with urlopen(urls["api"], timeout=12, context=context) as response:
        root = ET.fromstring(response.read())
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace) or "").split())
        published = (entry.findtext("atom:published", default="", namespaces=namespace) or "")[:10]
        authors = [author.findtext("atom:name", default="", namespaces=namespace) for author in entry.findall("atom:author", namespace)]
        link = next((node.attrib.get("href", "") for node in entry.findall("atom:link", namespace) if node.attrib.get("rel") == "alternate"), "")
        rows.append({"published": published, "title": title, "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""), "url": link, "summary": summary})
    return rows


class BibliographyDialog(QDialog):
    def __init__(self, object_name: str, parent=None) -> None:
        super().__init__(parent)
        self.object_name = object_name
        self.setWindowTitle(f"Bibliographic search · {object_name}")
        self.resize(820, 640)
        layout = QVBoxLayout(self)
        self.results = QTextBrowser()
        self.results.setOpenExternalLinks(True)
        self.results.setHtml(self._header("Searching arXiv for papers mentioning time series, periods, periodicity, timing, or variability…"))
        layout.addWidget(self.results, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _header(self, body: str) -> str:
        urls = bibliography_urls(self.object_name)
        return (
            f'<h2>Bibliographic Search</h2><p>Object searched: <b>{escape(self.object_name)}</b></p>'
            f'<p><a href="{escape(urls["ADS"], quote=True)}">Open ADS search</a> &nbsp;|&nbsp; '
            f'<a href="{escape(urls["arXiv"], quote=True)}">Open arXiv search</a></p>{body}'
        )

    def show_results(self, rows: list[dict[str, str]] | None = None, error: str = "") -> None:
        if error:
            body = f"<p>Automatic arXiv search is unavailable: {escape(error)}</p>"
        elif not rows:
            body = "<p>No automatic arXiv matches were returned. Use the ADS or arXiv search links above.</p>"
        else:
            items = []
            for row in rows:
                link = escape(row.get("url", ""), quote=True)
                title = escape(row.get("title", "Untitled"))
                title_html = f'<a href="{link}">{title}</a>' if link else title
                items.append(
                    f"<article><h3>{title_html}</h3><p><i>{escape(row.get('published', ''))} · {escape(row.get('authors', ''))}</i></p>"
                    f"<p>{escape(row.get('summary', ''))}</p></article><hr>"
                )
            body = "".join(items)
        self.results.setHtml(self._header(body))


class InputPage(QWidget):
    resolve_requested = Signal(str)
    bibliography_requested = Signal(str)
    send_requested = Signal(dict)
    input_changed = Signal()

    def __init__(self, state: WorkbenchState) -> None:
        super().__init__()
        self.state = state
        self._refreshing_state = False
        layout = QVBoxLayout(self)
        file_row = QHBoxLayout()
        self.file_label = QLabel("No light curve loaded")
        self.file_label.setObjectName("fileLabel")
        self.open_button = QPushButton("Open light curve…")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_file)
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.file_label, 1)
        layout.addLayout(file_row)

        column_row = QHBoxLayout()
        self.ascii_group = QGroupBox("ASCII / text columns")
        self.ascii_group.setMaximumWidth(270)
        ascii_form = QFormLayout(self.ascii_group)
        self.time_col = int_box(1, 1, 500)
        self.flux_col = int_box(2, 1, 500)
        self.error_col = int_box(3, 1, 500)
        for box in (self.time_col, self.flux_col, self.error_col):
            box.setMaximumWidth(82)
        self.use_error = QCheckBox("Use error column")
        self.use_error.setChecked(True)
        ascii_form.addRow("Time", self.time_col)
        ascii_form.addRow("Flux", self.flux_col)
        ascii_form.addRow("Error", self.error_col)
        ascii_form.addRow("", self.use_error)

        self.fits_panel = QGroupBox("FITS table columns")
        self.fits_panel.setMaximumWidth(310)
        fits_form = QFormLayout(self.fits_panel)
        self.fits_extension = QComboBox()
        self.fits_time = QComboBox()
        self.fits_flux = QComboBox()
        self.fits_error = QComboBox()
        self.fits_error.addItem("None", "")
        for combo in (self.fits_extension, self.fits_time, self.fits_flux, self.fits_error):
            combo.setMaximumWidth(150)
        self._configured_file_signature = None
        self.fits_extension.currentIndexChanged.connect(self.configure_fits_columns)
        fits_form.addRow("Extension", self.fits_extension)
        fits_form.addRow("Time", self.fits_time)
        fits_form.addRow("Flux", self.fits_flux)
        fits_form.addRow("Error", self.fits_error)
        column_row.addWidget(self.ascii_group)
        column_row.addWidget(self.fits_panel)
        column_row.addStretch(1)
        layout.addLayout(column_row)

        form = QFormLayout()
        self.flux_magnitude = QCheckBox("Flux is magnitude")
        self.time_unit = QComboBox()
        self.time_unit.addItems(["days", "seconds"])
        self.time_unit.setMaximumWidth(130)
        self.barycentric = QComboBox()
        self.barycentric.addItem("Auto-detect / approximate if possible", "auto")
        self.barycentric.addItem("Already barycentric (confirmed)", "already_barycentric")
        self.barycentric.addItem("Keep input times (demonstrative)", "uncorrected")
        self.barycentric.setMaximumWidth(310)
        form.addRow("", self.flux_magnitude)
        form.addRow("Time units", self.time_unit)
        form.addRow("Barycentric timing", self.barycentric)
        self.target_name = QLineEdit()
        self.target_name.setPlaceholderText("e.g. 4U 2206+54")
        self.target_name.setMaximumWidth(260)
        self.simbad_button = QPushButton("Search SIMBAD")
        self.simbad_button.setText("SIMBAD")
        self.simbad_button.setMaximumWidth(92)
        self.simbad_button.setToolTip("Resolve this object name with SIMBAD")
        self.simbad_button.clicked.connect(lambda: self.resolve_requested.emit(self.target_name.text().strip()))
        self.bibliography_button = QPushButton("Bibliography…")
        self.bibliography_button.setMaximumWidth(125)
        self.bibliography_button.setToolTip("Search ADS and arXiv for this object")
        self.bibliography_button.clicked.connect(lambda: self.bibliography_requested.emit(self.target_name.text().strip()))
        self.bibliography_button.setEnabled(False)
        self._resolved_target_name = ""
        self.target_name.textChanged.connect(self._target_name_changed)
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_name, 1)
        target_row.addWidget(self.simbad_button)
        target_row.addWidget(self.bibliography_button)
        self.target_ra = QLineEdit()
        self.target_dec = QLineEdit()
        self.target_ra.setMaximumWidth(150)
        self.target_dec.setMaximumWidth(150)
        form.addRow("Object name", target_row)
        form.addRow("RA [deg]", self.target_ra)
        form.addRow("Dec [deg]", self.target_dec)
        self.target_status = QLabel("")
        self.target_status.setObjectName("status")
        form.addRow("", self.target_status)
        self.limits = {name: QLineEdit() for name in ("xmin", "xmax", "ymin", "ymax")}
        for box in self.limits.values():
            box.setPlaceholderText("auto")
            box.setMaximumWidth(92)
        limit_row = QHBoxLayout()
        for name, box in self.limits.items():
            box.setToolTip(name)
            limit_row.addWidget(QLabel(name))
            limit_row.addWidget(box)
        form.addRow("Analysis limits", limit_row)
        layout.addLayout(form)

        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setLabel("bottom", "Time")
        self.raw_plot.setLabel("left", "Flux")
        self.raw_plot.showGrid(x=True, y=True, alpha=0.18)
        self.raw_plot.setClipToView(True)
        self.raw_plot.setDownsampling(auto=True, mode="peak")
        self._updating_plot_range = False
        self._initial_view_range: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.raw_plot.getViewBox().sigRangeChanged.connect(self._plot_range_changed)
        for selector in (self.time_col, self.flux_col, self.error_col):
            selector.valueChanged.connect(lambda _value: self.mark_plot_stale())
        self.use_error.toggled.connect(lambda _checked: self.mark_plot_stale())
        for selector in (self.fits_extension, self.fits_time, self.fits_flux, self.fits_error):
            selector.currentIndexChanged.connect(lambda _index: self.mark_plot_stale())
        plot_row = QHBoxLayout()
        self.plot_button = QPushButton("Plot selected columns")
        self.plot_button.clicked.connect(self.plot_selected_columns)
        self.plot_status = QLabel("Choose the columns, then plot the light curve.")
        self.plot_status.setObjectName("status")
        self.send_button = QPushButton("Send to frequency analysis")
        self.send_button.setObjectName("primary")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(lambda: self.send_requested.emit(self.collect_input_fields()))
        self.transfer_status = QLabel("Not sent to frequency analysis yet.")
        self.transfer_status.setObjectName("status")
        plot_row.addWidget(self.plot_button)
        plot_row.addWidget(self.send_button)
        plot_row.addWidget(self.plot_status, 1)
        layout.addLayout(plot_row)
        layout.addWidget(self.transfer_status)
        self.raw_plot_navigation = PlotNavigationBar(self.raw_plot)
        layout.addWidget(self.raw_plot_navigation)
        layout.addWidget(self.raw_plot)
        layout.addStretch(1)
        self._connect_input_changes()
        self.state.changed.connect(self.refresh_state)

    def _connect_input_changes(self) -> None:
        for widget in (self.time_col, self.flux_col, self.error_col):
            widget.valueChanged.connect(lambda _value: self._mark_input_pending())
        for widget in (self.use_error, self.flux_magnitude):
            widget.toggled.connect(lambda _checked: self._mark_input_pending())
        for widget in (self.time_unit, self.barycentric, self.fits_extension, self.fits_time, self.fits_flux, self.fits_error):
            widget.currentIndexChanged.connect(lambda _index: self._mark_input_pending())
        for widget in (self.target_name, self.target_ra, self.target_dec, *self.limits.values()):
            widget.textChanged.connect(lambda _text: self._mark_input_pending())

    def _mark_input_pending(self) -> None:
        if self._refreshing_state:
            return
        self._update_transfer_status()
        self.input_changed.emit()

    def _update_transfer_status(self) -> None:
        self.send_button.setEnabled(bool(self.state.file_bytes))
        committed = self.state.analysis_input_fields
        if not self.state.file_bytes:
            self.transfer_status.setText("Open a light curve to send it to Frequency Search.")
        elif committed is None:
            self.transfer_status.setText("Not sent to frequency analysis yet.")
        elif not self.input_matches_committed():
            self.transfer_status.setText("Input changed — send it again before analysis.")
        else:
            self.transfer_status.setText("Input sent to Frequency Search.")

    def input_matches_committed(self) -> bool:
        committed_input = self.state.analysis_input_fields
        return (
            committed_input is not None
            and bool(self.state.file_bytes)
            and self.collect_input_fields() == {key: committed_input[key] for key in INPUT_FIELD_KEYS if key in committed_input}
        )

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open light curve",
            "",
            "Light curves (*.txt *.dat *.lc *.fits *.fit *.fts *.ftz *.gz);;All files (*)",
        )
        if not path:
            return
        try:
            data = Path(path).read_bytes()
            engine.validate_upload(Path(path).name, data)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open file", str(exc))
            return
        same_file = self.state.filename == Path(path).name and self.state.file_bytes == data
        self.state.filename = Path(path).name
        self.state.file_bytes = data
        self.state.result = None
        self.state.analysis_input_fields = None
        if not same_file:
            self.state.sampling_window_cache = None
        self.configure_fits()
        self.state.fields = self.collect_input_fields()
        self.file_label.setText(f"{self.state.filename} · {len(data):,} bytes")
        self.state.notify()
        self._update_transfer_status()
        self.raw_plot.clear()
        self.raw_plot_navigation.clear_home_range()
        self._initial_view_range = None
        self.plot_status.setText("Choose the columns, then plot the light curve.")

    def configure_fits(self) -> None:
        is_fits = engine.is_fits_upload(self.filename_or_empty(), self.state.file_bytes)
        self.fits_panel.setEnabled(is_fits)
        self.ascii_group.setEnabled(not is_fits)
        self.fits_extension.clear()
        self.fits_time.clear()
        self.fits_flux.clear()
        self.fits_error.clear()
        self.fits_error.addItem("None", "")
        if not is_fits:
            return
        try:
            extensions = engine.fits_table_metadata(self.state.file_bytes)
        except Exception as exc:
            self.fits_panel.setToolTip(str(exc))
            return
        for extension in extensions:
            self.fits_extension.addItem(extension["label"], int(extension["index"]))
        self.configure_fits_columns()

    def filename_or_empty(self) -> str:
        return self.state.filename

    def configure_fits_columns(self) -> None:
        self.fits_time.clear()
        self.fits_flux.clear()
        selected = self.fits_extension.currentData()
        if selected is None:
            return
        try:
            extension = next(row for row in engine.fits_table_metadata(self.state.file_bytes) if int(row["index"]) == int(selected))
        except (StopIteration, ValueError):
            return
        names = [str(column["name"]) for column in extension["columns"]]
        for name in names:
            self.fits_time.addItem(name)
            self.fits_flux.addItem(name)
            self.fits_error.addItem(name, name)
        if names:
            self.fits_time.setCurrentText(names[0])
            self.fits_flux.setCurrentText(names[min(1, len(names) - 1)])

    def collect_input_fields(self) -> dict[str, str]:
        fields = {
            "time_col": str(self.time_col.value()),
            "flux_col": str(self.flux_col.value()),
            "error_col": str(self.error_col.value()) if self.use_error.isChecked() else "",
            "flux_is_magnitude": str(self.flux_magnitude.isChecked()).lower(),
            "time_unit": self.time_unit.currentText(),
            "barycentric_mode": str(self.barycentric.currentData()),
            "target_name": self.target_name.text().strip(),
            "target_ra_deg": self.target_ra.text().strip(),
            "target_dec_deg": self.target_dec.text().strip(),
        }
        fields.update({name: box.text().strip() for name, box in self.limits.items()})
        if engine.is_fits_upload(self.state.filename, self.state.file_bytes) and self.fits_extension.currentData() is not None:
            fields.update({
                "fits_extension": str(self.fits_extension.currentData()),
                "fits_time_col": self.fits_time.currentText(),
                "fits_flux_col": self.fits_flux.currentText(),
                "fits_error_col": str(self.fits_error.currentData() or ""),
            })
        return fields

    def apply_fields(self, fields: dict[str, str]) -> None:
        self.time_col.setValue(int(fields.get("time_col", 1)))
        self.flux_col.setValue(int(fields.get("flux_col", 2)))
        error_text = fields.get("error_col", "3")
        self.use_error.setChecked(bool(error_text))
        if error_text:
            self.error_col.setValue(int(error_text))
        self.flux_magnitude.setChecked(fields.get("flux_is_magnitude", "false").lower() == "true")
        self.time_unit.setCurrentText(fields.get("time_unit", "days"))
        barycentric_index = self.barycentric.findData(fields.get("barycentric_mode", "auto"))
        if barycentric_index >= 0:
            self.barycentric.setCurrentIndex(barycentric_index)
        self.target_name.setText(fields.get("target_name", ""))
        self.target_ra.setText(fields.get("target_ra_deg", ""))
        self.target_dec.setText(fields.get("target_dec_deg", ""))
        for name, box in self.limits.items():
            box.setText(fields.get(name, ""))
        if fields.get("fits_extension"):
            ext_index = self.fits_extension.findData(int(fields["fits_extension"]))
            if ext_index >= 0:
                self.fits_extension.setCurrentIndex(ext_index)
            self.fits_time.setCurrentText(fields.get("fits_time_col", ""))
            self.fits_flux.setCurrentText(fields.get("fits_flux_col", ""))
            self.fits_error.setCurrentText(fields.get("fits_error_col", "None"))

    def refresh_state(self) -> None:
        self._refreshing_state = True
        signature = (self.state.filename, len(self.state.file_bytes), hash(self.state.file_bytes))
        file_changed = signature != self._configured_file_signature
        if signature != self._configured_file_signature:
            self._configured_file_signature = signature
            self.configure_fits()
        self.file_label.setText(
            f"{self.state.filename} · {len(self.state.file_bytes):,} bytes" if self.state.filename else "No light curve loaded"
        )
        if self.state.fields:
            self.apply_fields(self.state.fields)
        self._update_bibliography_button()
        if file_changed:
            self.raw_plot.clear()
            self.raw_plot_navigation.clear_home_range()
            self._initial_view_range = None
            self.plot_status.setText("Choose the columns, then plot the light curve." if self.state.file_bytes else "Open a light curve to plot it.")
        self._refreshing_state = False
        self._update_transfer_status()

    def plot_selected_columns(self) -> None:
        if not self.state.file_bytes or not self.state.filename:
            QMessageBox.information(self, "No light curve", "Open a light-curve file first.")
            return
        fields = self.collect_input_fields()
        try:
            t, y, _dy = engine.read_columns(
                self.state.file_bytes,
                self.state.filename,
                int(fields["time_col"]) - 1,
                int(fields["flux_col"]) - 1,
                int(fields["error_col"]) - 1 if fields["error_col"] else None,
                fits_extension=int(fields["fits_extension"]) if fields.get("fits_extension") else None,
                fits_time_col=fields.get("fits_time_col") or None,
                fits_flux_col=fields.get("fits_flux_col") or None,
                fits_error_col=fields.get("fits_error_col") or None,
            )
        except Exception as exc:
            self.plot_status.setText("Could not plot selected columns.")
            QMessageBox.critical(self, "Could not plot light curve", str(exc))
            return
        self.raw_plot.clear()
        self.raw_plot.plot(t, y, pen=None, symbol="o", symbolSize=3, symbolBrush=VIU_ORANGE)
        self.raw_plot.setLabel("left", "Flux")
        self._updating_plot_range = True
        try:
            self.raw_plot_navigation.set_home_from_data(t, y)
            self._initial_view_range = tuple(
                tuple(float(value) for value in axis_range)
                for axis_range in self.raw_plot.viewRange()
            )
            range_args = {}
            for key, axis_name in (("xmin", "xRange"), ("ymin", "yRange")):
                upper_key = "xmax" if key == "xmin" else "ymax"
                try:
                    lower = float(self.limits[key].text())
                    upper = float(self.limits[upper_key].text())
                except ValueError:
                    continue
                if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                    range_args[axis_name] = (lower, upper)
            if range_args:
                self.raw_plot.getViewBox().setRange(**range_args, padding=0)
        finally:
            self._updating_plot_range = False
        self.plot_status.setText(f"Plotted {len(t):,} rows using the selected columns.")

    def _plot_range_changed(self, _view, ranges) -> None:
        if self._updating_plot_range or self._initial_view_range is None:
            return
        current = tuple(tuple(float(value) for value in axis_range) for axis_range in ranges)
        initial = self._initial_view_range

        def covers_initial(current_axis, initial_axis) -> bool:
            span = max(abs(initial_axis[1] - initial_axis[0]), 1e-15)
            tolerance = span * 0.005
            return current_axis[0] <= initial_axis[0] + tolerance and current_axis[1] >= initial_axis[1] - tolerance

        if covers_initial(current[0], initial[0]) and covers_initial(current[1], initial[1]):
            values = {name: "" for name in self.limits}
        else:
            values = {
                "xmin": f"{current[0][0]:.8g}",
                "xmax": f"{current[0][1]:.8g}",
                "ymin": f"{current[1][0]:.8g}",
                "ymax": f"{current[1][1]:.8g}",
            }
        for name, value in values.items():
            if self.limits[name].text() != value:
                self.limits[name].setText(value)

    def mark_plot_stale(self) -> None:
        if self._refreshing_state:
            return
        if not hasattr(self, "raw_plot"):
            return
        self.raw_plot.clear()
        self.raw_plot_navigation.clear_home_range()
        self._initial_view_range = None
        self._mark_input_pending()
        if self.state.file_bytes:
            self.plot_status.setText("Column selection changed; click Plot selected columns to update the graph.")

    def _target_name_changed(self, _text: str) -> None:
        if self._refreshing_state:
            return
        self._resolved_target_name = ""
        self.target_ra.clear()
        self.target_dec.clear()
        self._update_bibliography_button()
        self._mark_input_pending()

    def _update_bibliography_button(self) -> None:
        current_name = self.target_name.text().strip()
        self.bibliography_button.setEnabled(bool(current_name) and current_name == self._resolved_target_name)

    def set_resolving(self, active: bool) -> None:
        self.simbad_button.setEnabled(not active)
        self.simbad_button.setText("…" if active else "SIMBAD")
        if active:
            self.target_status.setText("Resolving object name…")

    def accept_resolved_target(self, requested_name: str, result: tuple[float, float, str]) -> None:
        if self.target_name.text().strip() != requested_name.strip():
            self.target_status.setText("The object name changed. Search SIMBAD again for the current name.")
            return
        ra_deg, dec_deg, resolved = result
        self._resolved_target_name = requested_name.strip()
        self.target_ra.setText(f"{ra_deg:.10f}")
        self.target_dec.setText(f"{dec_deg:.10f}")
        self._update_bibliography_button()
        self.target_status.setText(f"Resolved: {resolved}")


class FrequencyPage(QWidget):
    run_requested = Signal(dict)
    error_requested = Signal(dict)
    baseline_requested = Signal()

    def __init__(self, state: WorkbenchState) -> None:
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        self.status = QLabel("Upload a light curve in Input to begin.")
        self.status.setObjectName("status")
        outer.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)
        plots_panel = QWidget()
        plots_layout = QVBoxLayout(plots_panel)
        self.metrics = QLabel("Data points used: —     Baseline: —     Primary period: —     T0: —")
        self.ls_plot = pg.PlotWidget()
        self.ls_plot.setLabel("left", "Power")
        self.ls_plot.setLabel("bottom", "Frequency", units="cycles/day")
        self.ls_plot.showGrid(x=True, y=True, alpha=0.18)
        self.ls_plot.setClipToView(True)
        self.ls_plot.setDownsampling(auto=True, mode="peak")
        self.window_plot = pg.PlotWidget()
        self.window_plot.setLabel("left", "Window power")
        self.window_plot.setLabel("bottom", "Frequency", units="cycles/day")
        self.window_plot.showGrid(x=True, y=True, alpha=0.18)
        self.window_plot.setClipToView(True)
        self.window_plot.setDownsampling(auto=True, mode="peak")
        plots_layout.addWidget(self.metrics)
        plots_layout.addWidget(QLabel("Lomb–Scargle periodogram"))
        self.ls_navigation = PlotNavigationBar(self.ls_plot)
        plots_layout.addWidget(self.ls_navigation)
        plots_layout.addWidget(self.ls_plot, 2)
        plots_layout.addWidget(QLabel("Sampling window"))
        self.window_navigation = PlotNavigationBar(self.window_plot)
        plots_layout.addWidget(self.window_navigation)
        plots_layout.addWidget(self.window_plot, 1)
        split.addWidget(plots_panel)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        search_group = QGroupBox("Period search")
        form = QFormLayout(search_group)
        self.baseline_button = QPushButton("Subtract flux baseline")
        self.baseline_button.setToolTip("Subtract the weighted median flux baseline from the selected analysis data")
        self.baseline_button.clicked.connect(self.baseline_requested.emit)
        self.baseline_button.setEnabled(False)
        form.addRow("Data preparation", self.baseline_button)
        self.fmin = double_box(0.01, 0.0, 1e12, 8, 0.001)
        self.fmax = double_box(1.0, 0.0, 1e12, 8, 0.01)
        self.max_peaks = int_box(3, 1, 20)
        self.window_threshold = double_box(0.01, 0.0, 1e12, 5, 0.005)
        self.window_tolerance = double_box(0.01, 0.001, 1e12, 5, 0.001)
        self.samples_per_peak = double_box(10, 1, 1e6, 2, 1)
        self.min_period = double_box(2, 0, 1e12, 5, 0.1)
        self.exclusion_tolerance = double_box(0.015, 0.001, 1e12, 4, 0.001)
        for control in (
            self.fmin, self.fmax, self.max_peaks, self.window_threshold,
            self.window_tolerance, self.samples_per_peak, self.min_period,
            self.exclusion_tolerance,
        ):
            control.setMaximumWidth(128)
        self.manual_periods = QLineEdit()
        self.manual_periods.setMaximumWidth(180)
        form.addRow("Min frequency", self.fmin)
        form.addRow("Max frequency", self.fmax)
        form.addRow("Max considered peaks", self.max_peaks)
        form.addRow("Sampling-window threshold", self.window_threshold)
        form.addRow("Sampling-window tolerance", self.window_tolerance)
        form.addRow("Samples per peak", self.samples_per_peak)
        form.addRow("Min. considered period", self.min_period)
        form.addRow("Excluded periods", self.manual_periods)
        form.addRow("Manual exclusion tolerance", self.exclusion_tolerance)
        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.parameters()))
        form.addRow(self.run_button)
        controls_layout.addWidget(search_group)

        error_group = QGroupBox("Error estimation")
        error_form = QFormLayout(error_group)
        self.bootstrap_iterations = int_box(1000, 0, 1_000_000, 50)
        self.bootstrap_width = double_box(0.03, 0.001, 1e12, 4, 0.001)
        self.bootstrap_iterations.setMaximumWidth(128)
        self.bootstrap_width.setMaximumWidth(128)
        error_form.addRow("Bootstrap iterations", self.bootstrap_iterations)
        error_form.addRow("Bootstrap local width", self.bootstrap_width)
        self.error_button = QPushButton("Estimate errors")
        self.error_button.setToolTip("Estimate peak frequency and period errors with bootstrap resampling without rerunning Lomb–Scargle")
        self.error_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.error_button.clicked.connect(lambda: self.error_requested.emit(self.parameters()))
        error_form.addRow(self.error_button)
        controls_layout.addWidget(error_group)
        controls_layout.addStretch(1)
        split.addWidget(controls)
        controls.setMaximumWidth(340)
        split.setSizes([850, 330])

        self.peaks = QTableWidget(0, 7)
        self.peaks.setHorizontalHeaderLabels(["Label", "Frequency", "Freq. error", "Period", "Period error", "Power", "FAP"])
        self.peaks.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(QLabel("Detected peaks"))
        outer.addWidget(self.peaks)
        self.state.changed.connect(self.refresh_state)

    def parameters(self) -> dict[str, str]:
        return {
            "fmin": str(self.fmin.value()),
            "fmax": str(self.fmax.value()),
            "max_peaks": str(self.max_peaks.value()),
            "window_artifact_power": str(self.window_threshold.value()),
            "window_artifact_tolerance": str(self.window_tolerance.value()),
            "samples_per_peak": str(self.samples_per_peak.value()),
            "min_considered_period": str(self.min_period.value()),
            "n_bootstrap": str(self.bootstrap_iterations.value()),
            "bootstrap_width": str(self.bootstrap_width.value()),
            "manual_exclusions": self.manual_periods.text().strip(),
            "exclusion_tolerance": str(self.exclusion_tolerance.value()),
        }

    def apply_fields(self, fields: dict[str, str]) -> None:
        for widget, key in (
            (self.fmin, "fmin"), (self.fmax, "fmax"), (self.window_threshold, "window_artifact_power"),
            (self.window_tolerance, "window_artifact_tolerance"), (self.samples_per_peak, "samples_per_peak"),
            (self.min_period, "min_considered_period"), (self.bootstrap_width, "bootstrap_width"),
            (self.exclusion_tolerance, "exclusion_tolerance"),
        ):
            if fields.get(key):
                widget.setValue(float(fields[key]))
        for widget, key in ((self.max_peaks, "max_peaks"), (self.bootstrap_iterations, "n_bootstrap")):
            if fields.get(key):
                widget.setValue(int(float(fields[key])))
        self.manual_periods.setText(fields.get("manual_exclusions", fields.get("excluded_periods", "")))

    def refresh_state(self) -> None:
        if self.state.fields:
            self.apply_fields(self.state.fields)
        result = self.state.result
        self.run_button.setEnabled(self.state.analysis_input_fields is not None and self.state.file_bytes is not None)
        baseline_applied = bool((self.state.analysis_input_fields or {}).get("subtract_baseline", "false").lower() == "true")
        self.baseline_button.setEnabled(self.state.analysis_input_fields is not None)
        self.baseline_button.setText("Restore original flux" if baseline_applied else "Subtract flux baseline")
        self.baseline_button.setToolTip(
            "Restore the unshifted flux values for this input" if baseline_applied
            else "Subtract the weighted median flux baseline from the selected analysis data"
        )
        if not result:
            self.error_button.setEnabled(False)
            if not self.state.file_bytes:
                self.status.setText("Upload a light curve in Input to begin.")
            elif self.state.analysis_input_fields is None:
                self.status.setText("Send the selected input from Input before running the analysis.")
            else:
                self.status.setText("Ready to analyse locally.")
            self.ls_plot.clear()
            self.window_plot.clear()
            self.peaks.setRowCount(0)
            return
        unit = result.get("frequency_unit", "cycles/day")
        period_unit = result.get("period_unit", "d")
        primary = result.get("primary_period")
        self.metrics.setText(
            f"Data points used: {result.get('n_points', '—'):,}     Baseline: {result.get('baseline', 0):.5g} {result.get('baseline_unit', 'd')}"
            f"     Primary period: {float(primary):.6g} {period_unit}" if primary is not None else
            f"Data points used: {result.get('n_points', '—'):,}     Baseline: {result.get('baseline', 0):.5g} {result.get('baseline_unit', 'd')}     No usable primary period"
        )
        self.ls_plot.setLabel("bottom", "Frequency", units=unit)
        self.window_plot.setLabel("bottom", "Frequency", units=unit)
        series = result.get("series", {})
        self.error_button.setEnabled(bool(result.get("peaks")))
        self._set_line(self.ls_plot, series.get("frequency", []), series.get("power", []))
        self._set_line(self.window_plot, series.get("frequency", []), series.get("window_power", []))
        self.ls_navigation.set_home_from_data(series.get("frequency", []), series.get("power", []))
        self.window_navigation.set_home_from_data(series.get("frequency", []), series.get("window_power", []))
        peaks = result.get("peaks", [])
        self._add_peak_annotations(self.ls_plot, peaks, period_unit, prefix="LS")
        window_peaks = result.get("window_peaks", [])
        # Keep the sampling-window plot readable and lightweight when the grid
        # contains many local maxima; candidates are already ranked by power.
        self._add_peak_annotations(self.window_plot, window_peaks[:5], period_unit, color="#3B78A6", prefix="SW")
        self.peaks.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            values = (
                peak.get("label", ""),
                f"{float(peak.get('frequency', 0)):.7g} {unit}",
                f"{float(peak['frequency_error']):.3g} {unit}" if peak.get("frequency_error") is not None else "—",
                f"{float(peak.get('period', 0)):.7g} {period_unit}",
                f"{float(peak['period_error']):.3g} {period_unit}" if peak.get("period_error") is not None else "—",
                f"{float(peak.get('power', 0)):.5g}",
                f"{float(peak.get('fap', 0)):.3f}",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.peaks.setItem(row, col, item)
        if result.get("bootstrap_complete"):
            message = f"Error estimates from {int(result.get('bootstrap_iterations', 0)):,} bootstrap iterations."
        else:
            message = str(result.get("message") or "Analysis complete. Run Estimate errors to calculate bootstrap uncertainties.")
        if result.get("sampling_window_reused"):
            message += " Sampling window reused from the first run."
        elif len(series.get("window_power", [])) > 0:
            message += " Sampling window calculated for this input."
        self.status.setText(message)

    @staticmethod
    def _set_line(plot: pg.PlotWidget, x_values, y_values) -> None:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        plot.clear()
        if len(x) and len(y):
            plot.plot(x, y, pen=pg.mkPen(VIU_ORANGE, width=1.5))

    @staticmethod
    def _add_peak_annotations(
        plot: pg.PlotWidget,
        peaks: list[dict],
        period_unit: str,
        color: str = "#315B7D",
        prefix: str = "P",
    ) -> None:
        x_range, y_range = plot.viewRange()
        x_span = max(float(x_range[1] - x_range[0]), 1e-12)
        y_span = max(float(y_range[1] - y_range[0]), 1e-12)
        for index, peak in enumerate(peaks):
            try:
                frequency = float(peak["frequency"])
                power = float(peak["power"])
                period = float(peak.get("period", 1.0 / frequency))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if not np.isfinite(frequency) or not np.isfinite(power) or not np.isfinite(period):
                continue
            plot.addItem(
                pg.InfiniteLine(
                    pos=frequency,
                    angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                ),
                ignoreBounds=True,
            )
            plot.plot(
                [frequency], [power], pen=None, symbol="o", symbolSize=8,
                symbolBrush=color, symbolPen="w",
            )
            label = pg.TextItem(
                text=f"{prefix}{index + 1} · P={period:.4g} {period_unit}",
                color="#222330",
                anchor=(1.0, 0.5) if frequency > (x_range[0] + x_range[1]) / 2 else (0.0, 0.5),
                fill=pg.mkBrush(255, 255, 255, 230),
                border=pg.mkPen(color, width=1),
            )
            plot.addItem(label, ignoreBounds=True)
            horizontal_offset = x_span * 0.012
            x_position = frequency - horizontal_offset if frequency > (x_range[0] + x_range[1]) / 2 else frequency + horizontal_offset
            vertical_offset = y_span * 0.018 * (1 if index % 2 == 0 else -1)
            y_position = min(max(power + vertical_offset, y_range[0] + y_span * 0.015), y_range[1] - y_span * 0.015)
            label.setPos(x_position, y_position)


class FoldedPage(QWidget):
    update_requested = Signal(dict)

    def __init__(self, state: WorkbenchState) -> None:
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        controls = QFormLayout()
        self.t0 = QLineEdit()
        self.t0.setMaximumWidth(190)
        self.phase_bins = int_box(10, 4, 80)
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["harmonics", "selected"])
        self.fit_mode.setMaximumWidth(170)
        self.harmonics = int_box(1, 1, 8)
        self.periods = QLineEdit()
        self.periods.setPlaceholderText("Comma-separated periods")
        self.periods.setMaximumWidth(250)
        self.fit_method = QComboBox()
        self.fit_method.addItems(["standard", "robust", "display-optimized"])
        self.fit_method.setMaximumWidth(210)
        controls.addRow("T0", self.t0)
        controls.addRow("Phase bins", self.phase_bins)
        controls.addRow("Folded-fit frequencies", self.fit_mode)
        controls.addRow("Number of harmonics", self.harmonics)
        controls.addRow("Selected periods", self.periods)
        controls.addRow("Sinusoidal fitting", self.fit_method)
        self.update_button = QPushButton("Update folded profile")
        self.update_button.setObjectName("primary")
        self.update_button.clicked.connect(lambda: self.update_requested.emit(self.parameters()))
        controls.addRow(self.update_button)
        outer.addLayout(controls)
        self.summary = QLabel("Run a frequency analysis first.")
        outer.addWidget(self.summary)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Phase")
        self.plot.setLabel("left", "Flux")
        self.plot.setXRange(0, 1, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot_navigation = PlotNavigationBar(self.plot)
        outer.addWidget(self.plot_navigation)
        outer.addWidget(self.plot, 1)
        self.state.changed.connect(self.refresh_state)

    def parameters(self) -> dict[str, str]:
        return {
            "t0": self.t0.text().strip(),
            "fold_bins": str(self.phase_bins.value()),
            "fold_fit_mode": self.fit_mode.currentText(),
            "fold_fit_harmonics": str(self.harmonics.value()),
            "fold_fit_periods": self.periods.text().strip(),
            "model_fit_method": self.fit_method.currentText(),
        }

    def apply_fields(self, fields: dict[str, str]) -> None:
        self.t0.setText(fields.get("t0", ""))
        self.phase_bins.setValue(int(float(fields.get("fold_bins", 10))))
        self.fit_mode.setCurrentText(fields.get("fold_fit_mode", "harmonics"))
        self.harmonics.setValue(int(float(fields.get("fold_fit_harmonics", 1))))
        self.periods.setText(fields.get("fold_fit_periods", ""))
        self.fit_method.setCurrentText(fields.get("model_fit_method", "standard"))

    def refresh_state(self) -> None:
        if self.state.fields:
            self.apply_fields(self.state.fields)
        result = self.state.result
        self.plot.clear()
        if not result:
            self.plot_navigation.clear_home_range()
            self.summary.setText("Run a frequency analysis first.")
            return
        self.plot.setLabel("left", "Flux − baseline" if result.get("flux_baseline_subtracted") else "Flux")
        series = result.get("series", {})
        phase = np.asarray(series.get("fold_phase", []), dtype=float)
        flux = np.asarray(series.get("fold_flux", []), dtype=float)
        if len(phase) and len(flux):
            self.plot.plot(phase, flux, pen=None, symbol="o", symbolSize=5, symbolBrush=VIU_DARK)
        model_phase = np.asarray(series.get("fold_model_phase", []), dtype=float)
        model_flux = np.asarray(series.get("fold_model_flux", []), dtype=float)
        if len(model_phase) and len(model_flux):
            self.plot.plot(model_phase, model_flux, pen=pg.mkPen(VIU_ORANGE, width=2))
        home_phase = np.concatenate([values for values in (phase, model_phase) if len(values)]) if len(phase) or len(model_phase) else np.asarray([])
        home_flux = np.concatenate([values for values in (flux, model_flux) if len(values)]) if len(flux) or len(model_flux) else np.asarray([])
        self.plot_navigation.set_home_from_data(home_phase, home_flux, fixed_x_range=(0.0, 1.0))
        period = result.get("folded_period") or result.get("primary_period")
        self.summary.setText(
            f"Folding period: {float(period):.7g} {result.get('period_unit', 'd')} · T0: {float(result.get('t0', 0)):.7g}"
            if period is not None else "No usable period is available to fold."
        )


class EvidencePage(QWidget):
    EVIDENCE_FIELDS = [
        ("evidence_lightcurve_appearance", "1. Light-curve appearance, coverage, gaps, scatter, trends, and outliers"),
        ("evidence_bibliography_reviews", "2. Relevant references and short reviews"),
        ("evidence_frequency_reason", "3. Periodicity decision and selected-frequency justification"),
        ("evidence_frequency_harmonics", "3. Possible harmonics, aliases, and P versus 2P ambiguity"),
        ("evidence_folded_comments", "4. Folded-profile interpretation"),
        ("evidence_folded_harmonics", "4. Harmonics and ambiguity in the folded profile"),
        ("evidence_final_review", "5. Final assessment"),
        ("evidence_final_literature_comparison", "5. Comparison with the bibliography"),
    ]

    def __init__(self, state: WorkbenchState) -> None:
        super().__init__()
        self.state = state
        self.editors: dict[str, QTextEdit] = {}
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("Student evidence notebook"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        for key, label in self.EVIDENCE_FIELDS:
            layout.addWidget(QLabel(label))
            editor = QTextEdit()
            editor.setMinimumHeight(85)
            editor.setMaximumWidth(850)
            self.editors[key] = editor
            layout.addWidget(editor)
            editor.textChanged.connect(self.collect_evidence)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.state.changed.connect(self.refresh_state)

    def collect_evidence(self) -> None:
        self.state.evidence = {key: editor.toPlainText() for key, editor in self.editors.items()}

    def refresh_state(self) -> None:
        for key, editor in self.editors.items():
            text = self.state.evidence.get(key, "")
            if editor.toPlainText() != text:
                editor.blockSignals(True)
                editor.setPlainText(text)
                editor.blockSignals(False)


class SessionPage(QWidget):
    def __init__(self, state: WorkbenchState, main_window: "MainWindow") -> None:
        super().__init__()
        self.state = state
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Save the light curve, current analysis results, settings, and student evidence in one session file."))
        save_button = QPushButton("Save session…")
        save_button.clicked.connect(self.save)
        open_button = QPushButton("Restore session…")
        open_button.clicked.connect(self.open)
        row = QHBoxLayout()
        row.addWidget(save_button)
        row.addWidget(open_button)
        layout.addLayout(row)
        self.status = QLabel("Session data stays on this computer.")
        layout.addWidget(self.status)
        layout.addStretch(1)

    def save(self) -> None:
        self.main_window.sync_state_from_widgets()
        if not self.state.file_bytes:
            QMessageBox.information(self, "No light curve", "Open a light-curve file before saving a session.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "", "Workbench sessions (*.pwb)")
        if not path:
            return
        if not path.lower().endswith(".pwb"):
            path += ".pwb"
        try:
            save_session(path, self.state.snapshot())
        except Exception as exc:
            QMessageBox.critical(self, "Could not save session", str(exc))
            return
        self.status.setText(f"Saved: {Path(path).name}")

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Restore session", "", "Workbench sessions (*.pwb)")
        if not path:
            return
        try:
            snapshot = load_session(path)
            self.state.restore(snapshot)
        except Exception as exc:
            QMessageBox.critical(self, "Could not restore session", str(exc))
            return
        self.status.setText(f"Restored: {Path(path).name}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = WorkbenchState()
        self.worker: WorkThread | None = None
        self.bibliography_dialog: BibliographyDialog | None = None
        self.setWindowTitle(f"Periodicity Workbench {APP_VERSION}")
        self.resize(1420, 960)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.input_page = InputPage(self.state)
        self.frequency_page = FrequencyPage(self.state)
        self.folded_page = FoldedPage(self.state)
        self.evidence_page = EvidencePage(self.state)
        self.session_page = SessionPage(self.state, self)
        self.tabs.addTab(self.input_page, "Input")
        self.tabs.addTab(self.frequency_page, "Frequency Search")
        self.tabs.addTab(self.folded_page, "Folded Profile")
        self.tabs.addTab(self._advanced_placeholder(), "Advanced Analysis")
        self.tabs.addTab(self.evidence_page, "Evidence Notebook")
        self.tabs.addTab(self._report_placeholder(), "Report")
        self.tabs.addTab(self.session_page, "Session")
        self.frequency_page.run_requested.connect(self.run_analysis)
        self.frequency_page.error_requested.connect(self.estimate_errors)
        self.frequency_page.baseline_requested.connect(self.toggle_flux_baseline)
        self.folded_page.update_requested.connect(self.update_folded)
        self.input_page.resolve_requested.connect(self.start_target_resolution)
        self.input_page.bibliography_requested.connect(self.start_bibliography_search)
        self.input_page.send_requested.connect(self.send_input_to_analysis)
        self.input_page.input_changed.connect(self._input_draft_changed)
        self.state.changed.connect(self._input_draft_changed)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.statusBar().showMessage("Ready · analysis runs locally on this computer")

    @staticmethod
    def _advanced_placeholder() -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Iterative prewhitening, period tomography, and Model Laboratory are pending migration to the desktop interface."))
        layout.addStretch(1)
        return page

    @staticmethod
    def _report_placeholder() -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("PDF report generation is not yet connected to the desktop interface."))
        layout.addStretch(1)
        return page

    def collect_fields(self) -> dict[str, str]:
        fields = self.input_page.collect_input_fields()
        if self.state.analysis_input_fields is not None:
            fields["subtract_baseline"] = self.state.analysis_input_fields.get("subtract_baseline", "false")
        fields.update(self.frequency_page.parameters())
        fields.update(self.folded_page.parameters())
        if not fields.get("t0") and self.state.result:
            fields["t0"] = str(self.state.result.get("t0", ""))
        return fields

    def sync_state_from_widgets(self) -> None:
        self.state.fields = self.collect_fields()
        self.state.evidence = {key: editor.toPlainText() for key, editor in self.evidence_page.editors.items()}

    def _input_draft_changed(self) -> None:
        matches = self.input_page.input_matches_committed()
        self.frequency_page.run_button.setEnabled(matches)
        if not matches and not (self.worker is not None and self.worker.isRunning()):
            self.frequency_page.status.setText("Input has unsent changes. Return to Input and send them to Frequency Search.")

    def send_input_to_analysis(self, fields: dict[str, str]) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Wait for the current calculation to finish before sending new input.")
            return
        if not self.state.file_bytes or not self.state.filename:
            QMessageBox.information(self, "No light curve", "Open a light-curve file first.")
            return
        try:
            t, y, dy = engine.read_columns(
                self.state.file_bytes,
                self.state.filename,
                int(fields["time_col"]) - 1,
                int(fields["flux_col"]) - 1,
                int(fields["error_col"]) - 1 if fields.get("error_col") else None,
                fits_extension=int(fields["fits_extension"]) if fields.get("fits_extension") else None,
                fits_time_col=fields.get("fits_time_col") or None,
                fits_flux_col=fields.get("fits_flux_col") or None,
                fits_error_col=fields.get("fits_error_col") or None,
            )
            engine.apply_data_limits(t, y, dy, fields)
        except Exception as exc:
            QMessageBox.critical(self, "Could not send input", str(exc))
            return
        previous_input = self.state.analysis_input_fields or {}
        previous_fields = {key: previous_input[key] for key in INPUT_FIELD_KEYS if key in previous_input}
        changed = fields != previous_fields
        committed = dict(fields)
        committed["subtract_baseline"] = previous_input.get("subtract_baseline", "false")
        signature = sampling_window_signature(self.state.filename, committed)
        if self.state.sampling_window_cache is None or self.state.sampling_window_cache.get("signature") != signature:
            self.state.sampling_window_cache = None
        self.state.analysis_input_fields = committed
        if changed:
            self.state.result = None
        self.sync_state_from_widgets()
        self.state.notify()
        self.input_page._update_transfer_status()
        self.frequency_page.refresh_state()
        self.tabs.setCurrentWidget(self.frequency_page)
        self.statusBar().showMessage("Input sent to Frequency Search")

    def toggle_flux_baseline(self) -> None:
        if self.state.analysis_input_fields is None:
            QMessageBox.information(self, "Input not sent", "Send the selected data from Input before changing its baseline.")
            return
        fields = dict(self.state.analysis_input_fields)
        applied = fields.get("subtract_baseline", "false").lower() == "true"
        fields["subtract_baseline"] = "false" if applied else "true"
        self.state.analysis_input_fields = fields
        self.state.result = None
        self.sync_state_from_widgets()
        self.state.notify()
        if applied:
            self.frequency_page.status.setText("Original flux restored. Run analysis to refresh the results.")
            self.statusBar().showMessage("Original flux restored for Frequency Search")
        else:
            self.frequency_page.status.setText("Flux baseline subtraction applied to the analysis input. Run analysis to refresh the results.")
            self.statusBar().showMessage("Flux baseline subtraction applied")

    def _tab_changed(self, index: int) -> None:
        self.sync_state_from_widgets()
        page = self.tabs.widget(index)
        if page is self.frequency_page:
            self.frequency_page.refresh_state()
            self._input_draft_changed()
        elif page is self.folded_page:
            self.folded_page.refresh_state()

    def run_analysis(self, overrides: dict[str, str] | None = None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.state.analysis_input_fields is None:
            QMessageBox.information(self, "Input not sent", "Go to Input and press Send to frequency analysis first.")
            return
        if not self.input_page.input_matches_committed():
            self._input_draft_changed()
            QMessageBox.information(self, "Input changed", "Send the updated Input panel values to Frequency Search before running the analysis.")
            return
        if not self.state.file_bytes:
            QMessageBox.information(self, "No light curve", "Open a light-curve file in Input first.")
            return
        self.sync_state_from_widgets()
        fields = dict(self.state.analysis_input_fields)
        fields.update(self.frequency_page.parameters())
        fields.update(self.folded_page.parameters())
        if overrides:
            fields.update(overrides)
        if fields.get("manual_exclusions"):
            fields["excluded_periods"] = fields.pop("manual_exclusions")
        analysis_fields = dict(fields)
        analysis_fields["n_bootstrap"] = "0"
        window_signature = sampling_window_signature(self.state.filename, self.state.analysis_input_fields)
        window_cache = self.state.sampling_window_cache
        if window_cache is not None and window_cache.get("signature") != window_signature:
            window_cache = None
            self.state.sampling_window_cache = None
        self.frequency_page.run_button.setEnabled(False)
        self.frequency_page.error_button.setEnabled(False)
        if window_cache is None:
            self.frequency_page.status.setText("Running Lomb–Scargle and calculating the sampling window…")
        else:
            self.frequency_page.status.setText("Running Lomb–Scargle; reusing the cached sampling window…")
        self.statusBar().showMessage("Analysis running · the interface remains responsive")
        self.worker = WorkThread(
            engine.run_analysis,
            analysis_fields,
            self.state.file_bytes,
            self.state.filename,
            window_cache=window_cache,
            context=fields,
        )
        self.worker.succeeded.connect(self._analysis_finished)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _analysis_finished(self, result: dict, fields: dict[str, str]) -> None:
        self.state.fields = fields
        self.state.result = result
        if self.state.sampling_window_cache is None:
            series = result.get("series", {})
            frequency = series.get("frequency", [])
            window_power = series.get("window_power", [])
            if len(frequency) > 1 and len(window_power) == len(frequency):
                self.state.sampling_window_cache = {
                    "frequency": frequency,
                    "power": window_power,
                    "signature": sampling_window_signature(self.state.filename, self.state.analysis_input_fields or {}),
                }
        self.state.notify()
        self.tabs.setCurrentWidget(self.frequency_page)
        self.statusBar().showMessage("Analysis complete")

    def _analysis_failed(self, message: str) -> None:
        self.frequency_page.status.setText("Analysis failed")
        self.statusBar().showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis failed", message)

    def estimate_errors(self, overrides: dict[str, str] | None = None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.state.result or not self.state.result.get("peaks"):
            QMessageBox.information(self, "No analysis", "Run the frequency analysis before estimating errors.")
            return
        self.sync_state_from_widgets()
        fields = dict(self.state.fields)
        if overrides:
            fields.update(overrides)
        if int(float(fields.get("n_bootstrap", "1000"))) <= 1:
            QMessageBox.information(self, "Bootstrap iterations", "Set Bootstrap iterations to a value greater than 1.")
            return
        self.frequency_page.run_button.setEnabled(False)
        self.frequency_page.error_button.setEnabled(False)
        self.frequency_page.status.setText("Estimating peak errors with bootstrap resampling…")
        self.statusBar().showMessage("Error estimation running · the existing periodogram is being reused")
        self.worker = WorkThread(
            engine.estimate_bootstrap_errors,
            self.state.result,
            fields,
            context=fields,
        )
        self.worker.progress.connect(self._bootstrap_progress)
        self.worker.kwargs["progress_callback"] = self.worker.progress.emit
        self.worker.succeeded.connect(self._bootstrap_finished)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _bootstrap_progress(self, progress: dict) -> None:
        peak_count = int(progress.get("peak_count", 0))
        iteration = int(progress.get("iteration", 0))
        iterations = int(progress.get("iterations", 0))
        message = (
            f"Estimating errors · bootstrap {iteration:,}/{iterations:,} · "
            f"{peak_count} peaks per iteration"
        )
        self.frequency_page.status.setText(message)
        self.statusBar().showMessage(message)

    def _bootstrap_finished(self, result: dict, fields: dict[str, str]) -> None:
        self.state.result = result
        self.state.fields = fields
        self.state.notify()
        self.statusBar().showMessage("Bootstrap error estimation complete")

    def start_target_resolution(self, name: str) -> None:
        if not name:
            QMessageBox.information(self, "SIMBAD search", "Enter an object name first.")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Another calculation is still running.")
            return
        self.input_page.set_resolving(True)
        self.worker = WorkThread(resolve_target_name, name, context=("resolve", name))
        self.worker.succeeded.connect(self._async_action_finished)
        self.worker.failed.connect(self._async_action_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def start_bibliography_search(self, name: str) -> None:
        if not name:
            QMessageBox.information(self, "Bibliographic search", "Enter an object name first.")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Another calculation is still running.")
            return
        self.bibliography_dialog = BibliographyDialog(name, self)
        self.bibliography_dialog.show()
        self.worker = WorkThread(bibliography_search, name, context=("bibliography", name))
        self.worker.succeeded.connect(self._async_action_finished)
        self.worker.failed.connect(self._async_action_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _async_action_finished(self, result, context) -> None:
        action, name = context
        if action == "resolve":
            self.input_page.set_resolving(False)
            self.input_page.accept_resolved_target(name, result)
            self.sync_state_from_widgets()
        elif action == "bibliography" and self.bibliography_dialog is not None:
            self.bibliography_dialog.show_results(result)

    def _async_action_failed(self, message: str) -> None:
        context = self.worker.context if self.worker is not None else None
        action = context[0] if isinstance(context, tuple) else ""
        if action == "resolve":
            self.input_page.set_resolving(False)
            self.input_page.target_status.setText(f"Could not resolve: {message}")
            QMessageBox.warning(self, "Could not resolve target", message)
        elif action == "bibliography" and self.bibliography_dialog is not None:
            self.bibliography_dialog.show_results(error=message)
        else:
            self._analysis_failed(message)

    def _worker_finished(self) -> None:
        self.frequency_page.run_button.setEnabled(self.input_page.input_matches_committed())
        self.folded_page.update_button.setEnabled(True)
        self.frequency_page.error_button.setEnabled(bool(self.state.result and self.state.result.get("peaks")))
        self.worker = None

    def update_folded(self, parameters: dict[str, str]) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.state.result:
            QMessageBox.information(self, "No analysis", "Run Frequency Search first.")
            return
        self.sync_state_from_widgets()
        fields = dict(self.state.fields)
        fields.update(parameters)
        self.folded_page.update_button.setEnabled(False)
        self.statusBar().showMessage("Updating folded profile…")
        self.worker = WorkThread(engine.update_folded_profile, self.state.result, fields, context=fields)
        self.worker.succeeded.connect(self._folded_finished)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _folded_finished(self, result: dict, fields: dict[str, str]) -> None:
        self.state.result = result
        self.state.fields = fields
        self.state.notify()
        self.statusBar().showMessage("Folded profile updated")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Periodicity Workbench")
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QMainWindow { background: #F7F6F5; }"
        "QLabel { color: #222330; }"
        "QTabWidget::pane { border: 0; background: white; }"
        "QTabBar::tab { padding: 12px 15px; color: #6B6870; border-bottom: 2px solid transparent; }"
        "QTabBar::tab:selected { color: #E65014; border-bottom-color: #E65014; font-weight: 600; }"
        "QGroupBox { background: white; border: 1px solid #E4E1DF; border-radius: 8px; margin-top: 10px; padding: 10px; font-weight: 600; color: #222330; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 27px; border: 1px solid #D6D2D0; border-radius: 5px; padding: 2px 6px; background: white; color: #222330; }"
        "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #E65014; }"
        "QPushButton { padding: 6px 9px; border: 1px solid #D6D2D0; border-radius: 5px; background: white; color: #222330; }"
        "QPushButton:hover { background: #FFF1EB; border-color: #E65014; }"
        "QPushButton#primary { background: #E65014; border-color: #E65014; color: white; font-weight: 600; padding: 8px; border-radius: 5px; }"
        "QPushButton#primary:hover { background: #C94410; border-color: #C94410; }"
        "QPushButton:disabled { color: #9A9693; background: #F0EEED; border-color: #E4E1DF; }"
        "QToolButton { padding: 4px 7px; border: 1px solid transparent; border-radius: 4px; color: #4B4850; }"
        "QToolButton:hover { background: #FFF1EB; border-color: #E4E1DF; }"
        "QToolButton:checked { color: #E65014; background: #FFE1D3; border-color: #F1B49B; }"
        "QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #BEB9B6; border-radius: 3px; background: white; }"
        "QCheckBox::indicator:checked { background: #E65014; border-color: #E65014; }"
        "QLabel#status { color: #6B6870; padding: 4px; }"
        "QLabel#fileLabel { font-weight: 600; }"
        "QTableWidget { background: white; gridline-color: #E4E1DF; }"
        "QTableWidget::item:selected { background: #FFE1D3; color: #222330; }"
        "QHeaderView::section { background: #F7F6F5; color: #4B4850; border: 0; border-bottom: 1px solid #E4E1DF; padding: 6px; }"
    )
    pg.setConfigOptions(antialias=True, background="w", foreground=VIU_DARK)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
