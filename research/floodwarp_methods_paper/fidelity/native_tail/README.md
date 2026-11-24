# Discharge after rainfall in the incised catchment

Rainfall ended at hour six. Routed runoff can peak during recession. These budgets compare measured transport through two sections with changes in storage and infiltration between them. The downstream section is the open outlet. The upstream section spans the entire domain width.

## Water balance over the last 40 metres

| Method | Grid (m) | Window (h) | Outlet volume (m³) | Net volume entering reach (m³) | Storage released less infiltration plus rain (m³) | Reach contribution (%) |
|---|---:|---:|---:|---:|---:|---:|
| FV2 RK2 | 10 | 8–16 | 55,957.3 | 33,735.8 | 22,221.5 | 39.7 |
| FV2 RK2 | 5 | 8–12 | 37,141.5 | 35,075.6 | 2,065.9 | 5.6 |

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
| FV2 RK2 | 10 | 12 | 66.399 | 10:51 | unverified |
| FV2 RK2 | 5 | 12 | 48.330 | 4:48 | unverified |
| FV2 RK2 | 10 | 16 | 66.399 | 10:51 | yes |

[Rainfall and recession windows](crest_windows.csv) · [Rainfall and both discharge sections](incised_rain_and_routed_discharge.pdf)
