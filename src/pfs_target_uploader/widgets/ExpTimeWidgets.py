#!/usr/bin/env python3

import panel as pn
import param


class ExpTimeWidgets(param.Parameterized):
    def __init__(self):

        def _activate_exptime(event):
            if self.is_classical.value:
                self.single_exptime.disabled = False
            else:
                self.single_exptime.disabled = True
                self.single_exptime.value = 900

        self.single_exptime = pn.widgets.IntInput(
            # name="Individual Exposure Time (s) in [10, 7200]",
            value=900,
            step=10,
            start=10,
            end=7200,
            disabled=True,
        )
        # self.is_classical = pn.widgets.Checkbox(
        #     name="Classical-mode Observation", value=False
        # )
        self.is_classical = pn.widgets.Switch(
            name="Classical-mode Observation",
            value=False,
            width=25,
            align=("start", "center"),
        )

        i_activate_exptime = pn.bind(_activate_exptime, self.is_classical)

        self.pane = pn.Column(
            "### Individual Exposure Time (s)",
            pn.Row(
                self.is_classical,
                pn.pane.Markdown(
                    "<font size=3>Classical-mode Observation</font>",
                    align=("start", "center"),
                ),
            ),
            self.single_exptime,
            i_activate_exptime,
        )
