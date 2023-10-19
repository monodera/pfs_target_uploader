#!/usr/bin/env python3

import secrets
from datetime import datetime

import numpy as np
import pandas as pd
import panel as pn
import param
from logzero import logger
from zoneinfo import ZoneInfo

from .utils.checker import get_semester_daterange


class UploadNoteWidgets:
    # def __init__(self, message):
    def __init__(self, secret_token, uploaded_time):
        self.floatpanel = pn.layout.FloatPanel(
            None,
            # pn.pane.Markdown(message),
            name="Info",
            # config={"headerLogo": "<i class='fa-regular fa-thumbs-up fa-lg'></i>"},
            contained=False,
            position="center",
            # theme="none",
            theme="#3A7D7E",
            margin=20,
            width=720,
        )

        # JS on-click actions
        # https://github.com/awesome-panel/awesome-panel/blob/master/examples/js_actions.py
        # so far not working...
        stylesheet = """
        :host {
            --font-size: 2.5em;
            --color: darkcyan;
        }
        .bk-btn-light {
            color: darkcyan;
        }
        """

        self.copy_source_button = pn.widgets.Button(
            name=f"{secret_token}",
            icon="copy",
            # width=500,
            height=96,
            icon_size="1.5em",
            # button_style="outline",
            button_type="light",
            stylesheets=[stylesheet],
        )

        copy_source_code = "navigator.clipboard.writeText(source);"

        self.copy_source_button.js_on_click(
            args={"source": f"{secret_token}"},
            code=copy_source_code,
        )

        messages = [
            pn.pane.Markdown(
                "<i class='fa-regular fa-thumbs-up fa-2xl'></i><font size='4'>  Upload successful! Your **Upload ID** is the following.</font>"
            ),
            self.copy_source_button,
            pn.pane.Markdown(
                f"<font size='4'>Uploaded at {uploaded_time.isoformat(timespec='seconds')}</font>"
            ),
            pn.pane.Markdown(
                """
                - Please keep the Upload ID for the observation planning.
                - You can copy the Upload ID to the clipboard by clicking it.
                """
            ),
        ]

        self.floatpanel.objects = []
        for m in messages:
            self.floatpanel.objects.append(m)


class DocLinkWidgets:
    def __init__(self):
        self.doc = pn.pane.Markdown(
            "<font size='4'><i class='fa-solid fa-circle-info fa-lg' style='color: #3A7D7E;'></i> <a href='doc/index.html' target='_blank'>Documentation</a></font>",
            styles={"text-align": "right"},
        )
        self.pane = pn.Column(self.doc)


class FileInputWidgets(param.Parameterized):
    file_input = pn.widgets.FileInput(
        accept=".csv,.ecsv",
        multiple=False,
        sizing_mode="stretch_width",
    )
    secret_token = None

    pane = pn.Column(
        """# Step 1:
## Select a target list (<a href='doc/examples/example_targetlist.csv' target='_blank'>example</a>)""",
        file_input,
    )

    # FIXME:
    # - This watcher function does not work for browser engines other than Gecko.
    #   I don't know how to fix it...
    # - The function is called multiple times when selecting a file. I don't know why.
    @pn.depends("file_input.value", watch=True)
    def generate_secret_token(self):
        st = secrets.token_hex(8)
        logger.info(f"Secret Token Updated: {st}")
        logger.info(f"Filename: {self.file_input.filename}")
        logger.info(f"MIME Type: {self.file_input.mime_type}")
        self.secret_token = st

    # NOTE: When I put the `file_input` in `__init__(self)`, the watcher does not work.
    #       The reset function is a sort of a workaround.
    def reset(self):
        self.file_input.filename = None
        self.file_input.mime_type = None
        self.file_input.value = None


class StatusWidgets:
    def __init__(self, size=36):
        self.status_keys = pn.indicators.BooleanStatus(
            width=size, height=size, value=False, color="danger"
        )
        self.status_str = pn.indicators.BooleanStatus(
            width=size, height=size, value=False, color="danger"
        )
        self.status_vals = pn.indicators.BooleanStatus(
            width=size, height=size, value=False, color="danger"
        )
        self.status_flux = pn.indicators.BooleanStatus(
            width=size, height=size, value=False, color="danger"
        )
        self.status_dups = pn.indicators.BooleanStatus(
            width=size, height=size, value=False, color="danger"
        )

        self.status_grid = pn.GridBox(
            "<font size='3'>Stage 1</font>",
            "<font size='3'>Stage 2</font>",
            "<font size='3'>Stage 3</font>",
            "<font size='3'>Stage 3'</font>",
            "<font size='3'>Stage 4</font>",
            self.status_keys,
            self.status_str,
            self.status_vals,
            self.status_flux,
            self.status_dups,
            ncols=5,
            nrows=2,
            height=120,
        )

        """self.summary_nobj_L = pn.indicators.Number(
            name="Number of objects (low-resolution)",
            value=0,
            format="{value:d}",
            title_size="15pt",
            font_size="15pt",
            default_color="teal",
            visible=False,
        )

        self.summary_fh = pn.indicators.Number(
            name="Fiberhours",
            value=0,
            format="{value:.2f} h",
            title_size="15pt",
            font_size="15pt",
            default_color="teal",
        )#"""

        self.summary_table = pn.widgets.Tabulator(
            None,
            theme="bootstrap",
            theme_classes=["table-sm"],
            visible=False,
            layout="fit_data_table",
            hidden_columns=["index"],
            selectable=False,
            width=350,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": False}},
            disabled=True,
        )

        self.pane = pn.Column(
            self.status_grid,
            self.summary_table,
        )

    def reset(self):
        for s in [
            self.status_keys,
            self.status_str,
            self.status_vals,
            self.status_flux,
            self.status_dups,
        ]:
            s.value = False

        # self.summary_table.value = pd.DataFrame()
        self.summary_table.visible = False

    def show_results(self, df, validation_status):
        if validation_status["required_keys"]["status"] is None:
            pass
        elif validation_status["required_keys"]["status"]:
            self.status_keys.value = True
            if validation_status["optional_keys"]["status"]:
                self.status_keys.color = "success"
            else:
                self.status_keys.color = "warning"
        elif not validation_status["required_keys"]["status"]:
            self.status_keys.value = True
            self.status_keys.color = "danger"

        if validation_status["required_keys"]["status"]:
            if validation_status["str"]["status"] is None:
                pass
            elif validation_status["str"]["status"]:
                self.status_str.value = True
                self.status_str.color = "success"
            elif not validation_status["str"]["status"]:
                self.status_str.value = True
                self.status_str.color = "danger"

        if validation_status["str"]["status"]:
            if validation_status["values"]["status"] is None:
                pass
            elif validation_status["values"]["status"]:
                self.status_vals.value = True
                self.status_vals.color = "success"
            elif not validation_status["values"]["status"]:
                self.status_vals.value = True
                self.status_vals.color = "danger"

        if validation_status["values"]["status"]:
            if validation_status["flux"]["status"] is None:
                pass
            elif validation_status["flux"]["status"]:
                self.status_flux.value = True
                self.status_flux.color = "success"
            elif not validation_status["flux"]["status"]:
                self.status_flux.value = True
                self.status_flux.color = "danger"

        if validation_status["flux"]["status"]:
            if validation_status["unique"]["status"] is None:
                pass
            elif validation_status["unique"]["status"]:
                self.status_dups.value = True
                self.status_dups.color = "success"
            elif not validation_status["unique"]["status"]:
                self.status_dups.value = True
                self.status_dups.color = "danger"

        try:
            unique_priority = np.arange(0, 10, 1)  # np.unique(df["priority"])
            number_priority_L = np.zeros_like(unique_priority, dtype=int)
            number_priority_M = np.zeros_like(unique_priority, dtype=int)
            exptime_priority_L = np.zeros_like(unique_priority, dtype=float)
            exptime_priority_M = np.zeros_like(unique_priority, dtype=float)

            for i, p in enumerate(unique_priority):
                number_priority_L[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "L"), :
                ].index.size
                number_priority_M[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "M"), :
                ].index.size
                exptime_priority_L[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "L"), "exptime"
                ].sum()
                exptime_priority_M[i] = df.loc[
                    (df["priority"] == p) & (df["resolution"] == "M"), "exptime"
                ].sum()

            df_summary = pd.DataFrame(
                {
                    "priority": unique_priority,
                    "N_L": number_priority_L,
                    "Texp_L (FH)": exptime_priority_L / 3600,
                    "N_M": number_priority_M,
                    "Texp_M (FH)": exptime_priority_M / 3600,
                }
            )
            df_summary.loc[len(df_summary.index)] = [
                "total",
                sum(number_priority_L),
                sum(exptime_priority_L / 3600),
                sum(number_priority_M),
                sum(exptime_priority_M / 3600),
            ]

            self.summary_table.value = pd.DataFrame()
            self.summary_table.value = df_summary
            self.summary_table.visible = True

        except:
            pass


class TargetWidgets:
    def __init__(self):
        self.table_all = pn.widgets.Tabulator(
            None,
            page_size=50,
            theme="bootstrap",
            theme_classes=["table-striped", "table-sm"],
            frozen_columns=[],
            pagination="remote",
            header_filters=True,
            visible=False,
            layout="fit_data_table",
            disabled=True,
        )

        self.pane = pn.Column(self.table_all)

    def show_results(self, df):
        # it seems that frozen_columns must be empty when replacing its value
        self.table_all.frozen_columns = []
        self.table_all.value = pd.DataFrame()
        self.table_all.value = df
        self.table_all.frozen_columns = ["index"]
        self.table_all.visible = True

    def reset(self):
        # self.table_all.frozen_columns = []
        # self.table_all.value = pd.DataFrame()
        self.table_all.visible = False


class ResultWidgets:
    tabulator_kwargs = dict(
        page_size=50,
        theme="bootstrap",
        theme_classes=["table-striped", "table-sm"],
        frozen_columns=[],
        pagination="remote",
        header_filters=True,
        visible=False,
        layout="fit_data_table",
        disabled=True,
    )

    def __init__(self):
        # grand title of the main pane
        self.title = pn.pane.Markdown(
            """# Validation results
<font size='3'>Please check the validation results carefully and fix any errors to proceed to the submission.</font>
""",
            # renderer="myst",
            dedent=True,
        )

        # subsection titles
        self.error_title = pn.pane.Alert(
            """### Errors
Detected errors are listed below. Please fix them.
            """,
            alert_type="danger",
        )
        self.warning_title = pn.pane.Alert(
            """### Warnings
Detected warnings detected. Please take a look and fix them if possible and necessary.""",
            alert_type="warning",
        )
        self.info_title = pn.pane.Alert(
            """### Info""",
            alert_type="success",
        )

        # subsection texts
        self.error_text_success = pn.pane.Markdown("")
        self.error_text_keys = pn.pane.Markdown("")
        self.error_text_str = pn.pane.Markdown("")
        self.error_text_vals = pn.pane.Markdown("")
        self.error_text_flux = pn.pane.Markdown("")
        self.error_text_dups = pn.pane.Markdown("")

        self.warning_text_keys = pn.pane.Markdown("")
        self.warning_text_str = pn.pane.Markdown("")
        self.warning_text_vals = pn.pane.Markdown("")

        self.info_text_keys = pn.pane.Markdown("")
        self.info_text_str = pn.pane.Markdown("")
        self.info_text_vals = pn.pane.Markdown("")
        self.info_text_flux = pn.pane.Markdown("")
        self.info_text_dups = pn.pane.Markdown("")

        self.error_table_str = pn.widgets.Tabulator(None, **self.tabulator_kwargs)
        self.warning_table_str = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_vals = pn.widgets.Tabulator(None, **self.tabulator_kwargs)
        self.warning_table_vals = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_flux = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.error_table_dups = pn.widgets.Tabulator(None, **self.tabulator_kwargs)

        self.pane = pn.Column(
            self.title,
            # put errors first
            self.error_title,
            # show success message
            self.error_text_success,
            # errors on required keys (no need to put a table)
            self.error_text_keys,
            # errors on characters in string values
            self.error_text_str,
            self.error_table_str,
            # errors on out-of-range values
            self.error_text_vals,
            self.error_table_vals,
            # errors on flux columns
            self.error_text_flux,
            self.error_table_flux,
            # errors on duplicate ob_codes
            self.error_text_dups,
            self.error_table_dups,
            # warnings next
            self.warning_title,
            # warnings on optional keys
            self.warning_text_keys,
            # warnings on characters in string values (for optional keywords)
            self.warning_text_str,
            self.warning_table_str,
            # warnings on out-of-range values
            self.warning_text_vals,
            self.warning_table_vals,
            # successful vaildations last
            self.info_title,
            self.info_text_keys,
            self.info_text_str,
            self.info_text_vals,
            self.info_text_flux,
            self.info_text_dups,
            # height=200,
        )

    def reset(self):
        self.error_text_success.object = "\n####"
        self.error_text_keys.object = "\n####"
        self.error_text_str.object = "\n####"
        self.error_text_vals.object = "\n####"
        self.error_text_flux.object = "\n####"
        self.error_text_dups.object = "\n####"

        self.warning_text_keys.object = "\n####"
        self.warning_text_str.object = "\n####"
        self.warning_text_vals.object = "\n####"

        self.info_text_keys.object = "\n####"
        self.info_text_str.object = "\n####"
        self.info_text_vals.object = "\n####"
        self.info_text_flux.object = "\n####"
        self.info_text_dups.object = "\n####"

        for t in [
            self.error_table_str,
            self.error_table_dups,
            self.error_table_vals,
            self.error_table_flux,
            self.warning_table_str,
            self.warning_text_keys,
        ]:
            # t.frozen_columns = []
            # t.value = pd.DataFrame()
            t.visible = False

    def show_results(self, df, validation_status):
        # Stage 1 results
        for desc in validation_status["required_keys"]["desc_error"]:
            self.error_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_warning"]:
            self.warning_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["required_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        for desc in validation_status["optional_keys"]["desc_success"]:
            self.info_text_keys.object += f"\n<font size='3'>{desc}</font>\n"

        # Stage 2 results
        if validation_status["required_keys"]["status"]:
            if validation_status["str"]["status"] is None:
                pass
            elif validation_status["str"]["status"]:
                pass
            elif not validation_status["str"]["status"]:
                self.error_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
                self.error_table_str.frozen_columns = []
                self.error_table_str.value = pd.DataFrame()
                self.error_table_str.value = df.loc[
                    ~validation_status["str"]["success_required"], :
                ]
                self.error_table_str.frozen_columns = ["index"]
                self.error_table_str.visible = True

            if not validation_status["str"]["status_optional"]:
                self.warning_text_str.object += """\n
<font size='3'>String values must consist of `[A-Za-z0-9_-+.]`.</font>
<font size='3'>The following entries must be fixed.</font>
                """
                self.warning_table_str.frozen_columns = []
                self.warning_table_str.value = pd.DataFrame()
                self.warning_table_str.value = df.loc[
                    ~validation_status["str"]["success_optional"], :
                ]
                self.warning_table_str.frozen_columns = ["index"]
                self.warning_table_str.visible = True
        else:
            return

        # Stage 3 results
        if validation_status["str"]["status"]:
            if validation_status["values"]["status"] is None:
                pass
            elif validation_status["values"]["status"]:
                pass
            elif not validation_status["values"]["status"]:
                self.error_text_vals.object += """\n<font size='3'>Errors in values are detected for the following entries (See [documentation](doc/validation.html#stage-3)). </font>"""
                self.error_table_vals.frozen_columns = []
                self.error_table_vals.value = pd.DataFrame()
                self.error_table_vals.value = df.loc[
                    ~validation_status["values"]["success"], :
                ]
                self.error_table_vals.frozen_columns = ["index"]
                self.error_table_vals.visible = True
        else:
            return

        # Stage 3' results
        if validation_status["values"]["status"]:
            if validation_status["flux"]["status"]:
                self.info_text_flux.object += "\n<font size='3'>All ob_codes have at least one flux information</font>\n"
                self.error_table_flux.visible = False
            else:
                # add an error message and data table for duplicates
                self.error_text_flux.object += "\n<font size='3'>No flux info found in the following ob_codes:</font>\n"
                self.error_table_flux.frozen_columns = []
                self.error_table_flux.value = pd.DataFrame()
                self.error_table_flux.value = df.loc[
                    ~validation_status["flux"]["success"], :
                ]
                self.error_table_flux.frozen_columns = ["index"]
                self.error_table_flux.visible = True
        else:
            return

        # Stage 4 results
        if validation_status["flux"]["status"]:
            # logger.info(
            #     f"Status for duplicate: {validation_status['unique']['status']}"
            # )
            if validation_status["unique"]["status"]:
                self.info_text_dups.object += (
                    "\n<font size='3'>All `ob_code` are unique</font>\n"
                )
                # tweak for text update (I don't know the cause)
                # self.error_text_dups.object += "\n###"
                self.error_table_dups.visible = False
            else:
                # add an error message and data table for duplicates
                self.error_text_dups.object += "\n<font size='3'>`ob_code` must be unique within a proposal, but duplicate `ob_code` detected in the following targets:</font>\n"
                self.error_table_dups.frozen_columns = []
                self.error_table_dups.value = pd.DataFrame()
                self.error_table_dups.value = df.loc[
                    validation_status["unique"]["flags"], :
                ]
                self.error_table_dups.frozen_columns = ["index"]

                # BUG: it seems that the pandas-like styling does not work for panel>1.0.4 or so.
                # def _set_column_color(x, c="red"):
                #     print("setting background for the ob_code column")
                #     return [f"background-color: {c}" for _ in x]
                # self.table_duplicate.style.apply(
                #     _set_column_color,
                #     axis=0,
                #     subset=["ob_code"],
                #     # c="green",
                # )

                self.error_table_dups.visible = True
        else:
            return

        if (
            validation_status["required_keys"]["status"]
            and validation_status["str"]["status"]
            and validation_status["values"]["status"]
            and validation_status["flux"]["status"]
            and validation_status["unique"]["status"]
        ):
            self.error_text_success.object += "\n<font size='3'>No error is found. Congratulations. You can proceed to the submission.</font>\n"


class PPPresultWidgets:
    def __init__(self):
        self.ppp_title = pn.pane.Markdown(
            """# Results of PPP""",
            dedent=True,
        )

        self.ppp_figure = pn.Column("")

        self.pane = pn.Column(
            # "# Status",
            self.ppp_title,
            self.ppp_figure,
        )

    def reset(self):
        self.ppp_figure.clear()
        self.ppp_figure.visible = False

    def show_results(self, mode, nppc, p_result_fig, p_result_tab, ppp_Alert):
        self.ppp_figure.append(ppp_Alert)
        self.ppp_figure.append(
            pn.pane.Markdown(
                f"""## For the {mode:s} resolution mode:""",
                dedent=True,
            )
        )
        self.ppp_figure.append(nppc)
        self.ppp_figure.append(p_result_tab)
        self.ppp_figure.append(p_result_fig)
        self.ppp_figure.visible = True


class ValidateButtonWidgets:
    def __init__(self):
        self.validate = pn.widgets.Button(
            name="Validate",
            button_style="outline",
            button_type="primary",
            icon="stethoscope",
        )
        self.pane = pn.Column(
            """# Step 2:
## Check the uploaded list""",
            pn.Row(self.validate),
        )


class RunPppButtonWidgets:
    def __init__(self):
        self.PPPrun = pn.widgets.Button(
            name="Start (takes a few minutes ~ half hour)",
            button_style="outline",
            button_type="primary",
            icon="player-play-filled",
        )
        self.PPPrunStats = pn.Column("", width=100)

        self.pane = pn.Column(
            """# Step 3:
## Estimate the total required time""",
            self.PPPrun,
            self.PPPrunStats,
        )


class SubmitButtonWidgets:
    def __init__(self):
        self.submit = pn.widgets.Button(
            name="Submit Results",
            button_type="primary",
            icon="send",
            disabled=True,
        )
        self.pane = pn.Column(
            """# Step 4:
## Submit results""",
            self.submit,
        )


class DatePickerWidgets(param.Parameterized):
    def __init__(self):
        today = datetime.now(ZoneInfo("US/Hawaii"))

        semester_begin, semester_end = get_semester_daterange(
            today.date(),
            next=True,
        )

        self.date_begin = pn.widgets.DatePicker(
            name="Date Begin (HST)", value=semester_begin.date()
        )
        self.date_end = pn.widgets.DatePicker(
            name="Date End (HST)", value=semester_end.date()
        )

        self.pane = pn.Column("### Observation Period", self.date_begin, self.date_end)
