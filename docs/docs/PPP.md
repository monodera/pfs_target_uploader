# Total Time Estimate

The total exposure time required for an uploaded target list is estimated using the online PFS Pointing Planner (PPP).

## Run

- Press <u>"Start" button</u> to run the online PPP.

- The running time ranges from a few minutes to hours. It depends on the length of input target list. Please prevent uploading a huge list.

## Results

### 1. Status

PPP will give a status report of the outputs.

!!! danger "Warnings are raised in the following cases:"

    - the total requested time exceeds the 5-night upper limit for the normal progam;
    - some targets can not be completed in the semester due to their visibility, i.e., $\sum(T_\mathrm{observable})<T_\mathrm{request}$.

For example, 

<figure markdown>
  ![Status indicators](images/PPP_error.png){ width="600" }
  <!-- <figcaption>Status indicators</figcaption> -->
</figure>

If no warnings are reported, it will show 
<figure markdown>
  ![Status indicators](images/PPP_success.png){ width="800" }
  <!-- <figcaption>Status indicators</figcaption> -->
</figure>

### 2. Table

A table including the following information will be displayed, and its contents will be changed by dragging the slider above.

| Name                  |  Unit     | Description                                                                                        |
|-----------------------|-----------|----------------------------------------------------------------------------------------------------|
| resolution            |           | `low`, `medium` or `total`                                                                         |
| N_ppc                 |           | Number of pointings, can be adjusted by the slider                                                 |
| Texp                  | hour      | Total on-source time requested                                                                     |
| Texp                  | fiberhour | Total on-source time requested                                                                     |
| Request time1         | hour      | Total request time including overheads (calibration frames taken for each night)                   |
| Request time2         | hour      | Total request time including overheads (calibration frames taken for each fiber configuration)     |
| Used fiber fraction   |  %        | Average fiber usage fraction of pointings                                                          |
| Fraction of PPC <30%  |  %        | Fration of pointings having the fiber usage fraction < 30%                                         |
| P_all                 |  %        | Completion rate of the entire program                                                              |
| P_[1-9]               |  %        | Completion rate of each priority group                                                             |

- if only one resolution mode (low or medium) is requred, the table will only show information in that mode;
- if both modes are required, a third row `total` will be added

### 3. Figures

The <u>Completion Rate</u> (left), <u>Fiber Usage Fraction</u> (middle) and <u>Target Distribution</u> (right) will be shown.

- Completion Rate
    - title displays the resolution mode 
    - `PPC_id`: ID of PFS pointing center, pointings are sorted by the total priority of targets assigned on them
    - thick black solid line: completion rate of the entire program 
    - thick <span style="color: red;">red</span> solid line: completion rate of the primary sample (which has the smallest internal priority P) 
    - other lines: completion rate of each priority group   
    - vertical <span style="color: grey;">gray</span> dashed line: number of pointings required, can be adjusted by the slider above
    - <span style="color: orange;">orange</span> shade: area covered by Grade A programs in the last semester(s)
    - <span style="color: dodgerblue;">blue</span> shade: area covered by Grade B programs in the last semester(s)

- Fiber Usage Fraction
    - title displays the resolution mode 
    - `PPC_id`: ID of PFS pointing center, pointings are sorted by the total priority of targets assigned on them
    - thick <span style="color: red;">red</span> solid line: average fiber usage fraction of pointings  
    - vertical <span style="color: grey;">gray</span> dashed line: number of pointings required, can be adjusted by the slider above

- Target Distribution
    - targets in each priority group are plotted by different colors, with the primary sample (which has the smallest internal priority P) in red
    - transparent <span style="color: grey;">gray</span> hexagons show the pointings 