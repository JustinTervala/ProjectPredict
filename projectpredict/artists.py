from datetime import datetime
from math import sqrt, fabs

import networkx as nx


class ArtistBase(object):
    """Base class for artists. Contains methods to help determine the positions of the tasks

    Attributes:
        project (Project): The project to draw


    Args:
        project (Project): The project to draw
    """
    def __init__(self, project):
        self.project = project

    @staticmethod
    def _date_stat_to_timestamp(stat, attribute='latest_start'):
        return (getattr(stat, attribute)['mean'] - datetime.utcfromtimestamp(0)).total_seconds()

    def _find_optimal_distance(self, stats):
        """Finds the best distance between nodes.

        This is determined from the number of tasks in the longest path between all starting tasks and all
        terminal tasks. The optimal distance is the difference between the earliest latest finish date mean and the
        latest latest finish date mean divided by the number of nodes in the path.

        Args:
            stats (dict{Task: TaskStatistics}): The statistics used to derive the optimal distance

        Returns:
            float: The optimal distance between nodes.
        """
        start_tasks = []
        terminal_tasks = []
        for task in self.project.tasks:
            if len(list(self.project.predecessors(task))) == 0:
                start_tasks.append(task)
            if len(list(self.project.successors(task))) == 0:
                terminal_tasks.append(task)
        max_path, max_path_length = self._find_longest_path(start_tasks, terminal_tasks)
        greatest_time_difference = fabs(self._date_stat_to_timestamp(stats[max_path[1]])
                                        - self._date_stat_to_timestamp(stats[max_path[0]]))
        optimal_distance = greatest_time_difference / max_path_length
        return optimal_distance

    def _find_longest_path(self, start_tasks, terminal_tasks):
        max_path = (None, None)
        max_path_length = float('-inf')
        for start_task in start_tasks:
            for terminal_task in terminal_tasks:
                try:
                    path_length = max(
                        len(path)
                        for path in nx.all_shortest_paths(self.project, source=start_task, target=terminal_task))
                except nx.NetworkXNoPath:
                    continue
                if path_length > max_path_length:
                    max_path = (start_task, terminal_task)
                    max_path_length = path_length
        return max_path, max_path_length

    def _get_positions(self, stats):

        optimal_distance = self._find_optimal_distance(stats)

        task_generator = nx.topological_sort(self.project)
        first_task = next(task_generator)
        positions = {
            first_task: [(stats[first_task].latest_start['mean'] - datetime.utcfromtimestamp(0)).total_seconds(), 0]}
        for task in task_generator:
            x_position = (stats[task].latest_start['mean'] - datetime.utcfromtimestamp(0)).total_seconds()
            y_position = self._find_best_y_position(optimal_distance, positions, stats, task, x_position)
            positions[task] = [x_position, y_position]
        return positions

    def _find_best_y_position(self, optimal_distance, positions, stats, task, x_position):
        """
        The optimal Y position is found by first finding the best task for the new task to be positioned near and
        solving the equation for a circle centered at that task's position with a radius equal to the optimal_distance
        for the y-variable.
        """
        relevant_positions = {task_: pos for task_, pos in positions.items() if
                              fabs(pos[0] - x_position) <= optimal_distance}
        if relevant_positions:
            connected_tasks = {
                task_: pos for task_, pos in relevant_positions.items() if task_ in self.project.predecessors(task)}
            if connected_tasks:
                relevant_positions = connected_tasks

            max_task = self._find_best_neighbor_task(relevant_positions, stats)
            y_position = (
                    positions[max_task][1] +
                    sqrt(optimal_distance ** 2 - (x_position - positions[max_task][0]) ** 2))
        else:
            y_position = 0
        return y_position

    @staticmethod
    def _find_best_neighbor_task(relevant_positions, stats):
        max_start = float('-inf')
        max_task = None
        for task_, pos in relevant_positions.items():
            start = (stats[task_].latest_start['mean'] - datetime.utcfromtimestamp(0)).total_seconds()
            if start > max_start:
                max_start = start
                max_task = task_
        return max_task


class MatplotlibArtist(ArtistBase):
    """Draws a project using MatplotLib

    Note:
        There are still several issues with this artist. The task labels only fit a single letter, so the names often
        overflow. And the labels are too long and are improperly oriented.

    Attributes:
        project (Project): The project to draw


    Args:
        project (Project): The project to draw
    """
    def __init__(self, project):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcol
            import matplotlib.cm as cm
        except ImportError:
            print('Matplotlib must be installed to use the MatplotlibArtist')
            raise
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            print('Scipy must be installed to use the MatplotlibArtist')
            raise
        super(MatplotlibArtist, self).__init__(project)
        self.project = project
        self._plt = plt
        self._colors = mcol
        self._colormap = cm
        self._minimize = minimize_scalar

    def _get_color_converter(self, bounds, low_better, colormap):
        if low_better:
            colormap += '_r'
        scaled_colormap = self._colormap.get_cmap(colormap)
        color_norm = self._colors.Normalize(vmin=bounds[0], vmax=bounds[1])

        color_converter = self._colormap.ScalarMappable(norm=color_norm, cmap=scaled_colormap)
        color_converter.set_array([])
        return color_converter

    def draw(
            self,
            shade='total_float',
            stats=None,
            current_time=None,
            iterations=1000,
            colormap='Spectral',
            show_plot=True,
            show_variance=True,
            show_current_time=True):
        """Draws a project and shades it by derived stats.

        The X position of the tasks is determined by their latest start date

        Args:
            shade (str): Shades the nodes by a derived stat. Accepted values are 'total_float', 'latest_start', or
                'earliest_finish'
            stats (list[TaskStatistics], optional): The statistics used to draw the Project. If none are supplied, the
                Project will be sampled.
            current_time (datetime, optional): The current time to sample the Project. Only used if stats is not
                specified. Defaults to the current (UTC) time.
            iterations (int, optional): The number of iterations to sample the Project from. Only used if stats is not
                specified. Defaults to 1000
            colormap (str, optional): The matplotlib color map to use. Defaults to 'Spectral'
            show_plot (bool, optional): Show the plot? Defaults to True.
            show_variance (bool, optional): Show the variance of the latest start date? Defaults to True.
            show_current_time (bool, optional): Show the current time as a vertical line? Defaults to True.
        Returns:
            tuple: The figure and axis of the plot
        """
        allowed_shaders = ('total_float', 'latest_start', 'earliest_finish')
        if shade not in allowed_shaders:
            raise ValueError('shade {} not allowed. Allowed shaders are {}'.format(shade, allowed_shaders))
        stats = stats or self.project.calculate_task_statistics(current_time=current_time, iterations=iterations)

        color_converter, convert = self._create_color_converter(colormap, shade, stats)

        positions = self._get_positions(stats)

        fig, ax = self._plt.subplots()

        if show_variance:
            self._add_variance_bars(ax, positions, stats)

        if show_current_time:
            ax.axvline(x=(current_time - datetime.utcfromtimestamp(0)).total_seconds())

        for task in self.project.tasks:
            nx.draw_networkx_nodes(
                self.project,
                positions,
                ax=ax,
                nodelist=[task],
                node_color=color_converter.to_rgba(convert(getattr(stats[task], shade))))
        nx.draw_networkx_labels(self.project, positions, ax=ax, labels={task: task.name for task in self.project.tasks})
        nx.draw_networkx_edges(self.project, positions, ax=ax)

        self._adjust_ticks(ax)

        if show_plot:
            self._plt.show()
        return fig, ax

    @staticmethod
    def _adjust_ticks(ax):
        ax.set_xticklabels([str(datetime.utcfromtimestamp(tick)) for tick in ax.get_xticks()])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.grid(axis='x')
        ax.get_yaxis().set_visible(False)

    def _create_color_converter(self, colormap, shade, stats):
        min_max = (
            min(getattr(stat, shade) for stat in stats.values()),
            max(getattr(stat, shade) for stat in stats.values())
        )
        if shade == 'total_float':
            def convert(task_stat):
                return task_stat['mean'].total_seconds()
        else:
            def convert(task_stat):
                return (task_stat['mean'] - datetime.utcfromtimestamp(0)).total_seconds()
        min_max = [convert(val) for val in min_max]
        low_better = shade == 'latest_start'
        color_converter = self._get_color_converter(min_max, low_better, colormap)
        return color_converter, convert

    @staticmethod
    def _add_variance_bars(ax, positions, stats):
        bounds = {}
        for task, stat in stats.items():
            length = sqrt(stat.latest_start['variance'].total_seconds())
            scale = [positions[task][0] - length, positions[task][0] + length]
            bounds[task] = (scale, [positions[task][1]] * 2)
        for loc in bounds.values():
            ax.plot(loc[0], loc[1], zorder=-1, color='g')
