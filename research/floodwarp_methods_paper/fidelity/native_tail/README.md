# Discharge after rainfall in the incised catchment

Rainfall ended at hour six. Routed runoff can peak during recession. These budgets compare measured transport through two sections with changes in storage and infiltration between them. The downstream section is the open outlet. The upstream section spans the entire domain width.

## Water balance over the last 40 metres

| Method | Grid (m) | Window (h) | Outlet volume (m³) | Net volume entering reach (m³) | Storage released less infiltration plus rain (m³) | Reach contribution (%) |
|---|---:|---:|---:|---:|---:|---:|
| FV1 sparse | 5 | 8–16 | 82,965.6 | 81,615.1 | 1,350.5 | 1.6 |
| Hancock tiled | 5 | 8–16 | 37,473.1 | 35,115.6 | 2,357.5 | 6.3 |
| FV2 RK2 | 10 | 8–16 | 55,957.3 | 33,735.8 | 22,221.5 | 39.7 |
| FV2 RK2 | 5 | 8–16 | 37,363.7 | 35,075.6 | 2,288.0 | 6.1 |
| FV1 sparse | 10 | 8–16 | 89,565.3 | 86,794.2 | 2,771.1 | 3.1 |
| Hancock tiled | 10 | 8–16 | 56,852.6 | 33,741.0 | 23,111.6 | 40.7 |

Between hours 8 and 16, the 10 m minus 5 m FV2 RK2 difference in outlet volume was +18,594 m³. The upstream face supplied -1,340 m³ of this difference. Storage release in the reach, including net rain and infiltration, supplied +19,933 m³.

Between hours 8 and 16, the 10 m minus 5 m FV1 sparse difference in outlet volume was +6,600 m³. The upstream face supplied +5,179 m³ of this difference. Storage release in the reach, including net rain and infiltration, supplied +1,421 m³.

Between hours 8 and 16, the 10 m minus 5 m Hancock tiled difference in outlet volume was +19,379 m³. The upstream face supplied -1,375 m³ of this difference. Storage release in the reach, including net rain and infiltration, supplied +20,754 m³.

Each row uses the same run’s eight-hour checkpoint and latest saved endpoint. Rows with different endpoints describe different time windows. The source split for the whole window cannot establish the origin of a one-minute peak. Agreement between accepted volumes and storage changes checks conservation. The magnitude of the physical crest requires a separately qualified hydraulic reference.

[All section budgets](strip_budgets.csv) · [Boundary location controls](../native_outlet/README.md)

## Outlet maxima by observation horizon

Each row covers the interval from storm onset to the stated end time. Times refer to the end of the highest measured discharge interval. Only rows sharing an end time are directly comparable. A later maximum can arise from routed runoff, stored water or both.

| Method | Grid (m) | End (h) | Max interval mean (m³/s) | Peak interval ends (h:min) | Recession verified at end |
|---|---:|---:|---:|---:|---|
| FV1 sparse | 10 | 8 | 48.371 | 5:24 | unverified |
| FV1 sparse | 5 | 8 | 49.186 | 5:21 | unverified |
| FV2 RK2 | 10 | 8 | 47.475 | 4:45 | unverified |
| FV2 RK2 | 5 | 8 | 48.330 | 4:48 | unverified |
| Hancock tiled | 10 | 8 | 47.503 | 4:45 | unverified |
| Hancock tiled | 5 | 8 | 48.286 | 4:48 | unverified |
| FV1 sparse | 10 | 12 | 48.371 | 5:24 | unverified |
| FV1 sparse | 5 | 12 | 49.186 | 5:21 | unverified |
| FV2 RK2 | 10 | 12 | 66.399 | 10:51 | unverified |
| FV2 RK2 | 5 | 12 | 48.330 | 4:48 | unverified |
| Hancock tiled | 10 | 12 | 79.651 | 10:52 | unverified |
| Hancock tiled | 5 | 12 | 48.286 | 4:48 | unverified |
| FV1 sparse | 10 | 16 | 48.371 | 5:24 | yes |
| FV1 sparse | 5 | 16 | 49.186 | 5:21 | yes |
| FV2 RK2 | 10 | 16 | 66.399 | 10:51 | yes |
| FV2 RK2 | 5 | 16 | 48.330 | 4:48 | yes |
| Hancock tiled | 10 | 16 | 79.651 | 10:52 | yes |
| Hancock tiled | 5 | 16 | 48.286 | 4:48 | yes |

[Peaks at different observation intervals](interval_peaks.csv) · [Rainfall and recession windows](crest_windows.csv) · [Rainfall and both discharge sections](incised_rain_and_routed_discharge.pdf)

## Peak sensitivity to sampling through hour 16

The fixed intervals start at storm onset. The moving interval can end at any recorded minute. Both average accepted discharge volumes over the full stated duration.

| Method | Grid (m) | Interval (min) | Fixed peak (m³/s) | Moving peak (m³/s) |
|---|---:|---:|---:|---:|
| FV1 sparse | 10 | 1 | 48.37 | 48.37 |
| FV1 sparse | 10 | 5 | 48.36 | 48.36 |
| FV1 sparse | 10 | 15 | 48.29 | 48.29 |
| FV1 sparse | 5 | 1 | 49.19 | 49.19 |
| FV1 sparse | 5 | 5 | 49.16 | 49.18 |
| FV1 sparse | 5 | 15 | 49.09 | 49.10 |
| FV2 RK2 | 10 | 1 | 66.40 | 66.40 |
| FV2 RK2 | 10 | 5 | 47.44 | 51.17 |
| FV2 RK2 | 10 | 15 | 47.11 | 47.38 |
| FV2 RK2 | 5 | 1 | 48.33 | 48.33 |
| FV2 RK2 | 5 | 5 | 48.32 | 48.32 |
| FV2 RK2 | 5 | 15 | 48.16 | 48.24 |
| Hancock tiled | 10 | 1 | 79.65 | 79.65 |
| Hancock tiled | 10 | 5 | 47.48 | 59.25 |
| Hancock tiled | 10 | 15 | 47.15 | 47.41 |
| Hancock tiled | 5 | 1 | 48.29 | 48.29 |
| Hancock tiled | 5 | 5 | 48.28 | 48.28 |
| Hancock tiled | 5 | 15 | 48.12 | 48.20 |
