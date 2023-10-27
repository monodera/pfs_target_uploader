#!/usr/bin/env python3

import sys

import numpy as np
import panel as pn
from astropy import units as u
from astropy.table import Table
from logzero import logger

from ..utils.ppp import PPPrunStart, ppp_result


class PppResultWidgets:
    box_width = 1200

    # Maximum ROT can be requested for an openuse normal program
    max_reqtime_normal = 35.0

    def __init__(self):
        # PPP status
        # True if PPP has been run
        # False if PPP has not been run
        self.ppp_status = True

        self.ppp_title = pn.pane.Markdown(
            """# Results of PFS pointing simulation""",
            dedent=True,
            max_width=self.box_width,
        )

        self.ppp_warning = pn.pane.Alert(
            "<font size=5>⚠️ **Warnings**</font>\n\n"
            "<font size=3>The total requested time exceeds 35 hours (maximum for a normal program). "
            "Please make sure to adjust it to your requirement before proceeding to the submission. "
            "Note that targets observable in the input observing period are considered.</font>",
            alert_type="warning",
            max_width=self.box_width,
        )

        self.ppp_success = pn.pane.Alert(
            "<font size=5>✅ **Success**</font>\n\n"
            "<font size=3>The total requested time is reasonable for normal program. "
            "Note that targets observable in the input period are considered.</font>",
            alert_type="success",
            max_width=self.box_width,
        )

        self.ppp_figure = pn.Column()

        self.ppp_alert = pn.Column()

        self.pane = pn.Column(
            self.ppp_title,
            self.ppp_figure,
        )

    def reset(self):
        self.ppp_alert.clear()
        self.ppp_figure.clear()
        self.ppp_figure.visible = False
        self.ppp_status = False

    def show_results(self):  # , mode, nppc, p_result_fig, p_result_tab, ppp_Alert):
        logger.info("showing PPP results")

        def update_reqtime(df):
            rot = np.ceil(df.iloc[-1]["Request time (h)"] * 10.0) / 10.0
            if rot > self.max_reqtime_normal:
                c = "crimson"
            else:
                # c = "#007b43"
                c = "#3A7D7E"
            return {"value": rot, "default_color": c}

        # A number indicator showing the current total ROT
        self.reqtime = pn.indicators.Number(
            name="Your total request is",
            format="{value:.1f} <font size=18>h</font>",
            width=250,
            refs=pn.bind(update_reqtime, self.p_result_tab),
            styles={"text-align": "right"},
            margin=(0, 30, 0, 0),
        )

        self.ppp_figure.append(self.ppp_alert)
        self.ppp_figure.append(
            pn.pane.Markdown(
                f"""## For the {self.res_mode:s} resolution mode:""", dedent=True
            )
        )
        self.ppp_figure.append(
            pn.Row(self.reqtime, pn.Column(self.nppc, self.p_result_tab))
        )
        self.ppp_figure.append(self.p_result_fig)
        self.ppp_figure.visible = True

        size_of_ppp_figure = sys.getsizeof(self.p_result_fig) * u.byte
        logger.info(
            f"size of the ppp_figure object is {size_of_ppp_figure.to(u.kilobyte)}"
        )
        logger.info("showing PPP results done")

    def run_ppp(self, df, validation_status, weights=None):
        if weights is None:
            weights = [4.02, 0.01, 0.01]

        tb_input = Table.from_pandas(df)
        tb_visible = tb_input[validation_status["visibility"]["success"]]

        (
            uS_L2,
            cR_L,
            cR_L_,
            sub_l,
            obj_allo_L_fin,
            uS_M2,
            cR_M,
            cR_M_,
            sub_m,
            obj_allo_M_fin,
        ) = PPPrunStart(tb_visible, weights)

        (
            self.res_mode,
            self.nppc,
            self.p_result_fig,
            self.p_result_ppc,
            self.p_result_tab,
        ) = ppp_result(
            cR_L_, sub_l, obj_allo_L_fin, uS_L2, cR_M_, sub_m, obj_allo_M_fin, uS_M2
        )

        if (
            self.p_result_tab.value.iloc[-1]["Request time (h)"]
            > self.max_reqtime_normal
        ):
            self.ppp_alert.append(self.ppp_warning)
        else:
            self.ppp_alert.append(self.ppp_success)

        self.ppp_status = True