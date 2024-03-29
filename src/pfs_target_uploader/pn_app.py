#!/usr/bin/env python3

import os
from datetime import datetime, timezone
from io import BytesIO

import gurobipy
import numpy as np
import pandas as pd
import panel as pn
from astropy.table import Table
from dotenv import dotenv_values
from loguru import logger

from .utils.io import load_file_properties, load_input
from .utils.ppp import ppp_result_reproduce
from .widgets import (
    DatePickerWidgets,
    DocLinkWidgets,
    FileInputWidgets,
    PppResultWidgets,
    RunPppButtonWidgets,
    StatusWidgets,
    SubmitButtonWidgets,
    TargetWidgets,
    TimerWidgets,
    UploadNoteWidgets,
    ValidateButtonWidgets,
    ValidationResultWidgets,
)


def _toggle_buttons(buttons: list, disabled: bool = True):
    for b in buttons:
        b.disabled = disabled


def target_uploader_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if os.path.exists(config["OUTPUT_DIR"]):
        logger.info(f"{config['OUTPUT_DIR']} already exists.")
    else:
        os.makedirs(config["OUTPUT_DIR"])
        logger.info(f"{config['OUTPUT_DIR']} created.")

    template = pn.template.MaterialTemplate(
        # template = pn.template.BootstrapTemplate(
        title="PFS Target Uploader",
        # sidebar_width=400,
        sidebar_width=420,
        header_background="#3A7D7E",
        busy_indicator=None,
        favicon="docs/site/assets/images/favicon.png",
    )

    # setup panel components
    panel_doc = DocLinkWidgets()

    panel_input = FileInputWidgets()
    panel_validate_button = ValidateButtonWidgets()
    panel_status = StatusWidgets()
    panel_ppp_button = RunPppButtonWidgets()
    panel_submit_button = SubmitButtonWidgets()

    panel_dates = DatePickerWidgets()

    panel_timer = TimerWidgets()

    panel_results = ValidationResultWidgets()
    panel_targets = TargetWidgets()
    panel_ppp = PppResultWidgets()

    panel_input.reset()

    button_set = [
        panel_input.file_input,
        panel_validate_button.validate,
        panel_ppp_button.PPPrun,
    ]

    placeholder_floatpanel = pn.Column(height=0, width=0)

    # bundle panels in the sidebar
    sidebar_column = pn.Column(
        panel_input.pane,
        pn.Column(
            pn.Row("<font size=5>**Select an operation**</font>", panel_timer.pane),
            pn.Row(
                panel_validate_button.pane,
                panel_ppp_button.pane,
                panel_submit_button.pane,
                sizing_mode="stretch_width",
            ),
            margin=(10, 0, 0, 0),
        ),
        pn.Column(
            pn.Row("<font size=5>**Validation status**</font>"),
            panel_status.pane,
            margin=(10, 0, 0, 0),
        ),
    )

    sidebar_configs = pn.Column(panel_dates.pane)

    tab_sidebar = pn.Tabs(
        ("Home", sidebar_column),
        ("Config", sidebar_configs),
    )

    # bundle panel(s) in the main area
    tab_panels = pn.Tabs(
        ("Input list", panel_targets.pane),
        ("Validation", panel_results.pane),
        ("Pointing Simulation", panel_ppp.pane),
    )

    sidepanel_column = pn.Column(
        panel_doc.pane,
        tab_sidebar,
    )

    main_column = pn.Column(
        placeholder_floatpanel,
        tab_panels,
        margin=(30, 0, 0, 0),
    )

    # put them into the template
    # template.sidebar.append(panel_doc.pane)
    template.sidebar.append(sidepanel_column)
    template.main.append(main_column)

    tab_panels.visible = False

    # define on_click callback for the "validate" button
    def cb_validate(event):
        # disable the buttons and input file widget while validation
        _toggle_buttons(button_set, disabled=True)

        placeholder_floatpanel.objects = []

        tab_panels.visible = False

        panel_status.reset()
        panel_results.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        panel_timer.timer(True)

        validation_status, df_input, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

        _toggle_buttons(button_set, disabled=False)
        panel_timer.timer(False)

        if validation_status is None:
            return

        panel_status.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)
        panel_results.show_results(df_validated, validation_status)

        panel_ppp.df_input = df_validated
        panel_ppp.df_summary = panel_status.df_summary

        tab_panels.active = 1
        tab_panels.visible = True

        if validation_status["status"]:
            panel_submit_button.enable_button(panel_ppp.ppp_status)

    # define on_click callback for the "PPP start" button
    def cb_PPP(event):
        _toggle_buttons(button_set, disabled=True)
        panel_submit_button.submit.disabled = True

        placeholder_floatpanel.objects = []

        # reset some panels
        panel_status.reset()
        panel_ppp.reset()

        pn.state.notifications.clear()

        panel_timer.timer(True)

        validation_status, df_input_, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

        if validation_status is None:
            _toggle_buttons(button_set, disabled=False)
            panel_timer.timer(False)
            return

        if not validation_status["visibility"]["status"]:
            logger.error("No visible object is found")
            pn.state.notifications.error(
                "Cannot simulate pointing for 0 visible targets",
                duration=0,
            )
            _toggle_buttons(button_set, disabled=False)
            panel_timer.timer(False)
            return

        panel_status.show_results(df_validated, validation_status)
        panel_results.show_results(df_validated, validation_status)
        panel_targets.show_results(df_validated)

        tab_panels.active = 1
        tab_panels.visible = True

        try:
            panel_ppp.origname = panel_input.file_input.filename
            panel_ppp.origdata = panel_input.file_input.value
            panel_ppp.df_summary = panel_status.df_summary

            panel_ppp.run_ppp(df_validated, validation_status)
            panel_ppp.show_results()

            tab_panels.active = 2

            # enable the submit button only with the successful validation
            if validation_status["status"]:
                panel_submit_button.enable_button(panel_ppp.ppp_status)
                panel_submit_button.submit.disabled = False

        except gurobipy.GurobiError as e:
            pn.state.notifications.error(f"{str(e)}", duration=0)
            pass

        _toggle_buttons(button_set, disabled=False)
        panel_timer.timer(False)

    def cb_submit(event):
        panel_submit_button.submit.disabled = True

        placeholder_floatpanel.objects = []

        logger.info("Submit button clicked.")
        logger.info("Validation before actually writing to the storage")

        panel_timer.timer(True)

        # do the validation again and again (input file can be different)
        # and I don't know how to implement to return value
        # from callback to another function (sorry)
        validation_status, df_input, df_validated = panel_input.validate(
            date_begin=panel_dates.date_begin.value,
            date_end=panel_dates.date_end.value,
        )

        if (validation_status is None) or (not validation_status["status"]):
            logger.error("Validation failed for some reason")

            tab_panels.visible = False

            panel_status.reset()
            panel_results.reset()

            pn.state.notifications.clear()

            if validation_status is None:
                panel_timer.timer(False)
                return
            else:
                panel_status.show_results(df_validated, validation_status)
                panel_results.show_results(df_validated, validation_status)
                panel_targets.show_results(df_validated)
                tab_panels.visible = True
                panel_timer.timer(False)
                return

        panel_ppp.origname = panel_input.file_input.filename
        panel_ppp.origdata = panel_input.file_input.value
        panel_ppp.df_summary = panel_status.df_summary
        panel_ppp.upload_time = datetime.now(timezone.utc)
        panel_ppp.secret_token = panel_input.secret_token

        outdir, outfile_zip, _ = panel_ppp.upload(outdir_prefix=config["OUTPUT_DIR"])

        panel_notes = UploadNoteWidgets(
            panel_ppp.secret_token,
            panel_ppp.upload_time,
            panel_ppp.ppp_status,
            outdir.replace(config["OUTPUT_DIR"], "data/", 1),
            outfile_zip,
        )
        placeholder_floatpanel[:] = [panel_notes.floatpanel]

        panel_submit_button.submit.disabled = True
        panel_timer.timer(False)

    # set callback to the buttons
    panel_validate_button.validate.on_click(cb_validate)
    panel_ppp_button.PPPrun.on_click(cb_PPP)
    panel_submit_button.submit.on_click(cb_submit)

    app = template.servable()

    return app


#
# admin app
#
def list_files_app():
    config = dotenv_values(".env.shared")

    logger.info(f"config params from dotenv: {config}")

    if not os.path.exists(config["OUTPUT_DIR"]):
        logger.error(f"{config['OUTPUT_DIR']} not found")
        raise ValueError

    template = pn.template.VanillaTemplate(
        title="PFS Target & Proposal Lists",
        # collapsed_sidebar=True,
        # header_background="#3A7D7E",
        # header_background="#C71585",  # mediumvioletred
        header_background="#dc143c",  # crimson
        busy_indicator=None,
        favicon="docs/site/assets/images/favicon.png",
        # sidebar_width=400,
    )

    df_files_tgt_psl = load_file_properties(
        config["OUTPUT_DIR"],
        ext="ecsv",
    )

    psl_info_input = pn.widgets.FileInput(
        value=None,
        filename=None,
        accept=".csv",
        multiple=False,
        height=30,
    )

    psl_info = pn.Column(
        pn.Row("<font size=4>Please upload the proposal info:</font>", height=50),
        psl_info_input,
        height=150,
    )

    # range sliders for filtering
    slider_nobj = pn.widgets.EditableRangeSlider(
        name="N (ob_code)",
        start=np.floor(df_files_tgt_psl["n_obj"].min() / 10) * 10,
        end=np.ceil(df_files_tgt_psl["n_obj"].max() / 10) * 10,
        step=1,
    )
    slider_fiberhour = pn.widgets.EditableRangeSlider(
        name="Fiberhour (h)",
        start=np.floor(df_files_tgt_psl["Exptime_tgt (FH)"].min()),
        end=np.ceil(df_files_tgt_psl["Exptime_tgt (FH)"].max()),
        step=1,
    )

    slider_rot_l = pn.widgets.EditableRangeSlider(
        name="ROT (low, h)",
        start=np.floor(df_files_tgt_psl["Time_tot_L (h)"].min()),
        end=np.ceil(df_files_tgt_psl["Time_tot_L (h)"].max()),
        step=1,
    )
    slider_rot_m = pn.widgets.EditableRangeSlider(
        name="ROT (medium, h)",
        start=np.floor(df_files_tgt_psl["Time_tot_M (h)"].min()),
        end=np.ceil(df_files_tgt_psl["Time_tot_M (h)"].max()),
        step=1,
    )

    # setup panel components

    # Target & psl summary table

    """def execute_javascript(script):
        script = f'<script type="text/javascript">{script}</script>'
        js_panel.object = script
        js_panel.object = ""

    def open_panel_download(event):
        if event.column == "download":
            p_href = df_files_tgt["fullpath"][event.row].replace(
                config["OUTPUT_DIR"], "data", 1
            )
            # c.f. https://www.w3schools.com/jsref/met_win_open.asp
            script = f"window.open('{p_href}', '_blank')"
            execute_javascript(script)#"""

    def open_panel_magnify(event):
        if event.column == "magnify":
            table_ppc.clear()
            u_id = df_files_tgt_psl["Upload ID"][event.row]
            p_ppc = os.path.split(df_files_tgt_psl["fullpath_psl"][event.row])[0]
            table_ppc_t = Table.read(os.path.join(p_ppc, f"ppc_{u_id}.ecsv"))
            table_tgt_t = Table.read(os.path.join(p_ppc, f"target_{u_id}.ecsv"))
            table_psl_t = Table.read(os.path.join(p_ppc, f"psl_{u_id}.ecsv"))
            (
                nppc_fin,
                p_result_fig_fin,
                p_result_ppc_fin,
                p_result_tab,
            ) = ppp_result_reproduce(table_ppc_t, table_tgt_t, table_psl_t)

            def tab_ppc_save(p_result_ppc_fin, p_result_tab):
                Table.from_pandas(p_result_ppc_fin).write(
                    f"data/temp/TAC_ppc_{u_id}.ecsv",
                    format="ascii.ecsv",
                    delimiter=",",
                    overwrite=True,
                )
                Table.from_pandas(p_result_tab).write(
                    f"data/temp/TAC_psl_{u_id}.ecsv",
                    format="ascii.ecsv",
                    delimiter=",",
                    overwrite=True,
                )
                return f"data/temp/TAC_ppc_{u_id}.ecsv"

            if nppc_fin is not None:
                output_status = pn.pane.Markdown(
                    f"<font size=3>You are checking program: Upload id = {u_id} </font>",
                )

                fd = pn.widgets.FileDownload(
                    callback=pn.bind(tab_ppc_save, p_result_ppc_fin, p_result_tab),
                    filename=f"TAC_ppc_{u_id}.csv",
                    button_type="primary",
                    width=280,
                    height=40,
                )

                table_ppc.append(pn.Row(output_status, fd, width=750))

            else:
                output_status = pn.pane.Markdown(
                    f"<font size=3>You are checking program: Upload id = {u_id} (no PPP outputs) </font>",
                )

                table_ppc.append(pn.Row(output_status, width=750))

            table_ppc.append(
                pn.Row(
                    pn.Column(p_result_ppc_fin, width=700, height=1000),
                    pn.Column(nppc_fin, p_result_tab, p_result_fig_fin),
                )
            )

    def Table_files_tgt_psl(column_checkbox_):
        if psl_info_input.value is not None:
            df_psl_info = load_input(
                BytesIO(psl_info_input.value),
                format="csv",
            )[0]

            _df_files_tgt_psl = pd.merge(
                df_files_tgt_psl, df_psl_info, left_on="Upload ID", right_on="Upload ID"
            )

        else:
            _df_files_tgt_psl = df_files_tgt_psl

        _hidden_columns = list(
            set(list(_df_files_tgt_psl.columns)) - set(column_checkbox_)
        )

        _table_files_tgt_psl = pn.widgets.Tabulator(
            _df_files_tgt_psl,
            page_size=500,
            theme="bootstrap",
            # theme_classes=["table-striped", "table-sm"],
            theme_classes=["table-striped"],
            frozen_columns=["index"],
            pagination="remote",
            header_filters=True,
            buttons={"magnify": "<i class='fa-solid fa-magnifying-glass'></i>"},
            layout="fit_data_table",
            hidden_columns=_hidden_columns,
            disabled=True,
            selection=[],
            selectable="checkbox",
        )

        _table_files_tgt_psl.add_filter(slider_nobj, "n_obj")
        _table_files_tgt_psl.add_filter(slider_fiberhour, "t_exp")
        _table_files_tgt_psl.add_filter(slider_rot_l, "Time_tot_L (h)")
        _table_files_tgt_psl.add_filter(slider_rot_m, "Time_tot_M (h)")

        _table_files_tgt_psl.on_click(open_panel_magnify)

        return _table_files_tgt_psl

    column_checkbox = pn.widgets.MultiChoice(
        name="Columns to show",
        value=["Upload ID", "n_obj", "Time_tot_L (h)", "Time_tot_M (h)", "timestamp"],
        options=list(df_files_tgt_psl.columns) + ["proposal ID", "PI name", "rank"],
    )

    table_files_tgt_psl = pn.bind(Table_files_tgt_psl, column_checkbox)

    # Details of PPC
    table_ppc = pn.Column()

    # -------------------------------------------------------------------

    sidebar_column = pn.Column(
        psl_info, slider_nobj, slider_fiberhour, slider_rot_l, slider_rot_m
    )

    tab_panels = pn.Tabs(
        ("Program info", pn.Column(column_checkbox, table_files_tgt_psl)),
        ("PPC details", table_ppc),
    )

    # put them into the template
    template.sidebar.append(sidebar_column)
    template.main.append(tab_panels)

    app = template.servable()

    return app
