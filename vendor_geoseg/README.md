# GeoSeg source provenance

Source: https://github.com/WangLibo1995/GeoSeg (GPL-3.0; license included here).
Copied from the user's local GeoSeg source for this baseline restoration:

- `dice.py`, `functional.py`, `soft_ce.py`: original loss implementations.
- `../model/UNetFormerGeoSeg.py`: original single-input UNetFormer with auxiliary head.
- `../model/FTUNetFormerGeoSeg.py`: original single-input Swin v1 FTUNetFormer.

The user-modified RGB+DSM `Modified_FTUNetFormer` is not used.
`../baseline_geoseg.py` provides project-local integration, online patch sampling,
and a modern torch optimizer subclass implementing GeoSeg's Lookahead schedule.
