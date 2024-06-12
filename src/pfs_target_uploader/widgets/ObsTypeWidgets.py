import panel as pn
import param


class ObsTypeWidgets(param.Parameterized):
    stylesheet = """
        .bk-btn-primary {
            border-color: #3A7D7E !important;
            // border-color: #d2e7de !important;
        }

        .bk-btn-primary:hover, .bk-btn-primary.bk-active {
            color: #ffffff !important;
            background-color: #008899 !important;
        }
        """

    def __init__(self):

        # single exposure time widget
        self.single_exptime = pn.widgets.IntInput(
            # name="Individual Exposure Time (s) in [10, 7200]",
            value=900,
            step=10,
            start=10,
            end=7200,
            disabled=True,
        )

        #
        # observation type widget
        #
        # self.obs_type = pn.widgets.RadioButtonGroup(
        #     options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
        #     value="queue",
        #     button_style="outline",
        #     button_type="primary",
        #     stylesheets=[self.stylesheet],
        # )
        self.obs_type = pn.widgets.Select(
            options={"Queue": "queue", "Classical": "classical", "Filler": "filler"},
            value="Queue",
        )

        self.obstype_pane = pn.Column(self.obs_type)

        self.exptime_pane = pn.Column(
            "### Individual Exposure Time (s)", self.single_exptime
        )
