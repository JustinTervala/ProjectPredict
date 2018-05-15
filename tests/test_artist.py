from __future__ import division

import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mock
import pytest

from projectpredict import Project, Task, TimeUnits
from projectpredict.artists import *
from tests.util import MockModel, make_task_stat


@pytest.fixture
def large_project():
    project = Project('proj', MockModel())
    tasks = {str(i): Task(str(i), data={'a': i}) for i in range(1, 9)}

    # Longest path will be {1, 2}-3-4-6-{7, 8}
    project.add_dependencies([
        (tasks['1'], tasks['3']),
        (tasks['2'], tasks['3']),
        (tasks['3'], tasks['4']),
        (tasks['4'], tasks['5']),
        (tasks['4'], tasks['6']),
        (tasks['6'], tasks['7']),
        (tasks['6'], tasks['8'])
    ])
    return project, tasks

@pytest.fixture
def large_project_with_stats():
    project, tasks = large_project()
    current_date = datetime(year=2018, month=5, day=15)
    vars = [1, 2, 3]
    stats = {
        tasks['1']: make_task_stat(current_date, [1, 2, 3], vars, TimeUnits.hours),
        tasks['2']: make_task_stat(current_date, [2, 3, 4], vars, TimeUnits.hours),
        tasks['3']: make_task_stat(current_date, [3, 4, 5], vars, TimeUnits.hours),
        tasks['4']: make_task_stat(current_date, [4, 5, 6], vars, TimeUnits.hours),
        tasks['5']: make_task_stat(current_date, [5, 6, 7], vars, TimeUnits.hours),
        tasks['6']: make_task_stat(current_date, [6, 7, 8], vars, TimeUnits.hours),
        tasks['7']: make_task_stat(current_date, [7, 8, 9], vars, TimeUnits.hours),
        tasks['8']: make_task_stat(current_date, [8, 9, 10], vars, TimeUnits.hours)
    }
    return project, tasks, stats


def test_artist_base_init():
    class MockProject: pass

    project = MockProject()
    artist = ArtistBase(project)
    assert artist.project is project


def test_artist_base_find_longest_path_length(large_project):
    project = large_project[0]
    start_tasks, terminal_tasks = project.get_starting_and_terminal_tasks()
    assert ArtistBase(project)._find_longest_path_length(start_tasks, terminal_tasks) == 5


def test_artist_base_find_longest_path_length_no_path():
    project = Project('proj', MockModel())
    tasks = {str(i): Task(str(i), data={'a': i}) for i in range(1, 4)}
    project.add_tasks(tasks.values())
    project.add_dependency(tasks['1'], tasks['2'])
    start_tasks, terminal_tasks = project.get_starting_and_terminal_tasks()
    assert ArtistBase(project)._find_longest_path_length(start_tasks, terminal_tasks) == 2


def test_artist_base_find_optimal_distance(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    earliest_task_stat_ls = stats[tasks['1']].latest_start
    latest_task_stat_ls = stats[tasks['8']].latest_start
    greatest_distance = (ArtistBase._date_to_timestamp(latest_task_stat_ls['mean'])
                         - ArtistBase._date_to_timestamp(earliest_task_stat_ls['mean']))
    optimal_distance = greatest_distance / 5
    artist = ArtistBase(project)

    assert artist._find_optimal_distance(stats) == optimal_distance


def test_artist_base_find_best_neighbor_task(large_project):
    project, tasks = large_project
    relevant_positions = {
        tasks['1']: ([1], [2]),
        tasks['2']: ([3], [4])
    }
    assert ArtistBase._find_best_neighbor_task(relevant_positions) == tasks['2']


def test_artist_base_get_relevant_positions_none_connected(large_project):
    project, tasks = large_project
    task = Task('a', data={'a': 3})
    positions = {
        tasks['1']: (1, 2),
        tasks['2']: (3, 4),
        tasks['3']: (4, 5)
    }
    project.add_task(task)
    artist = ArtistBase(project)
    relevant_positions = artist._get_relevant_positions(task, positions, 3.5, 1)
    assert relevant_positions == {tasks['2']: (3, 4), tasks['3']: (4, 5)}


def test_artist_base_get_relevant_positions_some_connected(large_project):
    project, tasks = large_project
    positions = {
        tasks['1']: (1, 2),
        tasks['2']: (3, 4),
        tasks['3']: (4, 5)
    }
    artist = ArtistBase(project)
    relevant_positions = artist._get_relevant_positions(tasks['4'], positions, 3.5, 1)
    assert relevant_positions == {tasks['3']: (4, 5)}


def test_artist_base_get_relevant_positions_none_near(large_project):
    project, tasks = large_project
    positions = {
        tasks['1']: (1, 2),
        tasks['2']: (3, 4),
        tasks['3']: (4, 5)
    }
    artist = ArtistBase(project)
    relevant_positions = artist._get_relevant_positions(tasks['4'], positions, 10, 1)
    assert relevant_positions == {}


def test_artist_base_calculate_y_position():
    best_pos = (3, 4)
    x_pos = 5
    optimal_dist = 2
    expected = best_pos[1] + sqrt(optimal_dist ** 2 - (x_pos - best_pos[0]) ** 2)
    assert ArtistBase._calculate_y_position(best_pos, x_pos, optimal_dist) == expected


def test_artist_base_find_best_y_position_none_near(large_project):
    project, tasks = large_project
    positions = {
        tasks['1']: (1, 2),
        tasks['2']: (3, 4),
        tasks['3']: (4, 5)
    }
    artist = ArtistBase(project)
    y_pos = artist._find_best_y_position(1, positions, tasks['4'], 10)
    assert y_pos == 0


def test_artist_base_find_best_y_position_none_connected(large_project):
    project, tasks = large_project
    task = Task('a', data={'a': 3})
    positions = {
        tasks['1']: (1, 2),
        tasks['2']: (3, 4),
        tasks['3']: (4, 5)
    }
    project.add_task(task)
    optimal_dist = 1
    x_pos = 3.5
    artist = ArtistBase(project)
    best_task_pos = (4, 5)
    expected_y_pos = ArtistBase._calculate_y_position(best_task_pos, x_pos, optimal_dist)

    y_pos = artist._find_best_y_position(optimal_dist, positions, task, x_pos)
    assert y_pos == expected_y_pos


def test_artist_base_get_positions(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    artist = ArtistBase(project)

    optimal_dist = artist._find_optimal_distance(stats)
    task_positions = artist.get_positions(stats)
    for task, pos in task_positions.items():
        assert pos[0] == artist._date_to_timestamp(stats[task].latest_start['mean'])
        for task2, pos2 in task_positions.items():
            if task2 is not task:
                distance = sqrt((pos[0] - pos2[0])**2 + (pos[1] - pos2[1])**2)
                assert distance >= optimal_dist


def test_matplotlib_artist_init():
    project = Project('proj')
    artist = MatplotlibArtist(project)
    assert artist.project is project


def test_matplotlib_artist_get_color_converter():
    project = Project('proj')
    artist = MatplotlibArtist(project)
    bounds = (3, 12)
    color_converter = artist._get_color_converter(bounds, False, 'binary')
    assert color_converter.to_rgba(3) == (1.0, 1.0, 1.0, 1.0)
    assert color_converter.to_rgba(12) == (0.0, 0.0, 0.0, 1.0)
    middle_color = color_converter.to_rgba(7.5)
    assert middle_color[0] == middle_color[1]
    assert middle_color[1] == middle_color[2]
    assert middle_color[0] - 0.5 <= 0.1
    assert middle_color[3] == 1.0


def test_matplotlib_artist_get_color_converter_low_better():
    project = Project('proj')
    artist = MatplotlibArtist(project)
    bounds = (3, 12)
    color_converter = artist._get_color_converter(bounds, True, 'binary')
    assert color_converter.to_rgba(12) == (1.0, 1.0, 1.0, 1.0)
    assert color_converter.to_rgba(3) == (0.0, 0.0, 0.0, 1.0)
    middle_color = color_converter.to_rgba(7.5)
    assert middle_color[0] == middle_color[1]
    assert middle_color[1] == middle_color[2]
    assert middle_color[0] - 0.5 <= 0.1
    assert middle_color[3] == 1.0


def test_matplotlib_artist_create_color_converter_total_float(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    colormap, convert = artist._create_color_converter('binary', 'total_float', stats)
    assert convert(stats[tasks['1']].total_float) == stats[tasks['1']].total_float['mean'].total_seconds()
    assert colormap.to_rgba(convert(stats[tasks['1']].total_float)) == (1.0, 1.0, 1.0, 1.0)
    assert colormap.to_rgba(convert(stats[tasks['8']].total_float)) == (0.0, 0.0, 0.0, 1.0)


def test_matplotlib_artist_create_color_converter_latest_start(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    colormap, convert = artist._create_color_converter('binary', 'latest_start', stats)
    assert convert(stats[tasks['1']].latest_start) == ArtistBase._date_to_timestamp(stats[tasks['1']].latest_start['mean'])
    assert colormap.to_rgba(convert(stats[tasks['1']].latest_start)) == (1.0, 1.0, 1.0, 1.0)
    assert colormap.to_rgba(convert(stats[tasks['8']].latest_start)) == (0.0, 0.0, 0.0, 1.0)


def test_matplotlib_artist_create_color_converter_earliest_finish(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    colormap, convert = artist._create_color_converter('binary', 'earliest_finish', stats)
    assert convert(stats[tasks['1']].earliest_finish) == ArtistBase._date_to_timestamp(stats[tasks['1']].earliest_finish['mean'])
    assert colormap.to_rgba(convert(stats[tasks['1']].earliest_finish)) == (0.0, 0.0, 0.0, 1.0)
    assert colormap.to_rgba(convert(stats[tasks['8']].earliest_finish)) == (1.0, 1.0, 1.0, 1.0)


def test_matplotlib_artist_adjust_ticks():
    _fig, ax = plt.subplots()
    current_time = datetime(year=2018, month=5, day=12)
    ticks = [ArtistBase._date_to_timestamp(current_time + timedelta(days=i)) for i in range(5)]
    ax.set_xticks(ticks)
    MatplotlibArtist._adjust_ticks(ax)
    for real_tick, expected_tick in zip(ax.get_xticklabels(), ticks):
        assert real_tick.get_text() == str(datetime.utcfromtimestamp(expected_tick))
        assert real_tick.get_rotation() == 45.0
    assert not ax.get_yaxis().get_visible()
    assert len(ax.get_xgridlines()) == len(ticks)


def test_matplotlib_artist_add_variance_bars(large_project_with_stats):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    positions = artist.get_positions(stats)
    bounds = {}
    for task, stat in stats.items():
        length = sqrt(stat.latest_start['variance'].total_seconds())
        scale = [positions[task][0] - length, positions[task][0] + length]
        bounds[task] = (scale, [positions[task][1]] * 2)
    _fig, ax = plt.subplots()
    MatplotlibArtist._add_variance_bars(ax, positions, stats)
    assert len(ax.lines) == len(bounds)
    expected_positons = [pos for pos in bounds.values()]
    color = ax.lines[0].get_color()
    for line in ax.lines:
        assert (list(line.get_xdata()), list(line.get_ydata())) in expected_positons
        assert line.get_color() == color
        assert line.get_zorder() == -1


def test_matplotlib_artist_draw_invalid_shade(large_project_with_stats):
    project, _tasks, _stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    with pytest.raises(ValueError):
        artist.draw(shade='invalid')


def test_matplotlib_artist_draw(large_project_with_stats, mocker):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    fig, ax = plt.subplots()
    mocker.patch.object(artist.project, 'calculate_task_statistics', return_value=stats)
    mock_subplots = mocker.patch.object(artist._plt, 'subplots', return_value=(fig, ax))
    mock_var = mocker.patch.object(artist, '_add_variance_bars')
    mock_vline = mocker.patch.object(ax, 'axvline')
    mock_show = mocker.patch.object(artist._plt, 'show')
    current_time = datetime(year=2018, month=6, day=3)
    ret_fig, ret_ax = artist.draw(current_time=current_time)
    mock_subplots.assert_called_once()
    mock_var.assert_called_once()
    mock_vline.assert_called_once_with(x=ArtistBase._date_to_timestamp(current_time))
    mock_show.assert_called()
    assert ret_fig == fig
    assert ret_ax == ax


def test_matplotlib_artist_draw_less(large_project_with_stats, mocker):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    fig, ax = plt.subplots()
    mocker.patch.object(artist.project, 'calculate_task_statistics', return_value=stats)
    mock_subplots = mocker.patch.object(artist._plt, 'subplots', return_value=(fig, ax))
    mock_var = mocker.patch.object(artist, '_add_variance_bars')
    mock_vline = mocker.patch.object(ax, 'axvline')
    mock_show = mocker.patch.object(artist._plt, 'show')
    current_time = datetime(year=2018, month=6, day=3)
    ret_fig, ret_ax = artist.draw(
        current_time=current_time,
        show_current_time=False,
        show_plot=False,
        show_variance=False)
    mock_subplots.assert_called_once()
    mock_var.assert_not_called()
    mock_vline.assert_not_called()
    mock_show.assert_not_called()
    assert ret_fig == fig
    assert ret_ax == ax

def test_matplotlib_artist_draw_with_stats(large_project_with_stats, mocker):
    project, tasks, stats = large_project_with_stats
    artist = MatplotlibArtist(project)
    fig, ax = plt.subplots()
    mock_stats = mocker.patch.object(artist.project, 'calculate_task_statistics', return_value=stats)
    mock_subplots = mocker.patch.object(artist._plt, 'subplots', return_value=(fig, ax))
    mock_var = mocker.patch.object(artist, '_add_variance_bars')
    mock_vline = mocker.patch.object(ax, 'axvline')
    mock_show = mocker.patch.object(artist._plt, 'show')
    current_time = datetime(year=2018, month=6, day=3)
    ret_fig, ret_ax = artist.draw(
        current_time=current_time,
        stats=stats,
        show_current_time=False,
        show_plot=False,
        show_variance=False)
    mock_stats.assert_not_called()
    mock_subplots.assert_called_once()
    mock_var.assert_not_called()
    mock_vline.assert_not_called()
    mock_show.assert_not_called()
    assert ret_fig == fig
    assert ret_ax == ax


def test_matplotlib_artist_init_import_error(mocker):
    project = Project('proj')
    with mock.patch.dict(sys.modules, {'matplotlib': None}):
        with pytest.raises(ImportError):
            MatplotlibArtist(project)