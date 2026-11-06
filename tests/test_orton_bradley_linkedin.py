"""Chronology and held native snapshots for the social-video profile."""
import numpy as np
import pytest

from examples.orton_bradley.render_linkedin import schedule_frames


@pytest.mark.parametrize('duration,expected_count', [(45.,1350),(8.,240)])
def test_native_timeline_includes_peaks_and_final_hold(duration, expected_count):
    saved=np.arange(0.,259201.,300.)
    mandatory=[0.,86700.,88200.,259200.]
    rows=schedule_frames(saved_times=saved,duration=duration,fps=30,
        control_video=[0.,6.,31.,42.,45.],control_simulation=[0.,64800.,108000.,259200.,259200.],
        mandatory_times=mandatory)
    times=np.array([row['time_s'] for row in rows])
    assert len(rows)==expected_count
    assert np.all(np.diff(times)>=0.)
    assert all(value in times for value in mandatory)
    assert np.all(np.isin(times,saved))
    assert all(row['time_s']==259200. for row in rows if row['production_time_s']>=42.)
    assert rows[0]['time_s']==0.
    if duration==45.:
        assert sum(times==259200.)==90
        assert rows[180]['time_s']==64800.
        assert rows[930]['time_s']==108000.


def test_missing_mandatory_snapshot_is_rejected():
    with pytest.raises(ValueError,match='Mandatory native snapshot is missing'):
        schedule_frames(saved_times=np.array([0.,300.,600.]),duration=1.,fps=30,
            control_video=[0.,1.],control_simulation=[0.,600.],mandatory_times=[450.])


@pytest.mark.parametrize('duration,count',[(20.,600),(6.,180)])
def test_short_edit_shows_both_native_peaks_early(duration,count):
    saved=np.arange(0.,259201.,300.)
    mandatory=[0.,72000.,86700.,88200.,172800.,259200.]
    rows=schedule_frames(saved_times=saved,duration=duration,fps=30,
        control_video=[0.,1.5,4.,10.5,13.5,18.,20.],
        control_simulation=[0.,64800.,82800.,90000.,108000.,259200.,259200.],
        mandatory_times=mandatory)
    times=np.array([row['time_s'] for row in rows])
    assert len(rows)==count and np.all(np.diff(times)>=0.)
    assert all(value in times for value in mandatory)
    assert np.all(np.isin(times,saved))
    for peak in (86700.,88200.):
        first=next(row for row in rows if row['time_s']==peak)
        assert 4.<first['production_time_s']<10.
    assert all(row['time_s']==259200. for row in rows if row['production_time_s']>=18.)
    if duration==20.:
        assert sum(times==259200.)==60
