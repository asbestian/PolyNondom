#!/usr/bin/env python3

"""Module for enumerating and visualising non-dominated points.

This module comprises classes for different feasible domains, a class for
multiple objectives and a class for enumerating and visualising 
different subsets of (non-dominated) points.

Available classes
-----------------

:class:`GenericDomain`
   represents the feasible domain of a(n abstract) generic optimisation problem.

:class:`AssignmentDomain`
   represents the feasible domain of an assignment problem.

:class:`CubeDomain`
   represents the feasible domain given by the vertices of a standard cube.

:class:`ExplicitDomain`
   represents the feasible domain given by an explicitly given set of feasible solutions.

:class:`Objectives`
   represents the objectives of a multi-criteria optimisation problem.

:class:`PolyNondom`
   enumerates and visualises different sets of (non-dominated) points.
"""

from argparse import ArgumentParser
from collections import Iterable
from itertools import combinations, permutations, product
import logging
import math
from numpy import array, dot, linspace, meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Error(Exception):
    """Base class for exceptions."""

class InfeasibleBoxError(Error):
    """Rectangular box is infeasible."""

class GenericDomain:
    """Feasible domain of a generic optimisation problem.
     
    :ivar int dim: Dimension of feasible domain
    
    .. note:: Do not use this class directly.
    """

    def __init__(self, dim):
        """Initialises generic domain.
        
        :param: int dim: Dimension of feasible domain
        """
        self.dim = dim

    def __iter__(self):
        """Iterator for generic domain."""
        raise NotImplementedError

    def __str__(self):
        """String representation of domain."""
        return self.__class__.__name__ + str(set([elem for elem in self]))


class AssignmentDomain(GenericDomain):
    """Feasible domain of an assignment problem.
    
    The assignment problem has a number of agents and a(n equal) number 
    of tasks. Any agent can be assigned to perform any task. A feasible 
    solution is given by an assignment of agents to tasks in such a way 
    that each agent is one task and all tasks are assigned by exactly 
    by one agent.
    
    :ivar int num_agents: Number of agents of corresponding assignment problem

    :Example: ad = AssignmentDomain(4)
    """

    def __init__(self, agents):
        """Initialises assignment domain."""
        super().__init__(agents**2)
        self.num_agents = agents

    def __iter__(self):
        """Iterator for assignment domain."""
        for perm in permutations(list(range(self.num_agents))):
            feas_sol = [0]*self.dim
            for i, val in enumerate(perm):
                feas_sol[i*self.num_agents + val] = 1
            yield tuple(feas_sol)


class CubeDomain(GenericDomain):
    """Feasible domain given by the vertices of an n-dimensional standard cube.
    
    :Example: cd = CubeDomain(3)
    """

    def __init__(self, dim):
        """Initialises feasible domain given by standard cube."""
        super().__init__(dim)

    def __iter__(self):
        """Iterator for cube domain."""
        for prod in product(range(2), repeat=self.dim):
            yield prod


class ExplicitDomain(GenericDomain):
    """Explicitely given feasible domain.
    
    :ivar Iterable domain: Feasible domain
    
    :Example: ed = ExplicitDomain(3, some_iterable)
    """

    def __init__(self, dim, domain):
        """Initialises feasible domain via given domain."""
        super().__init__(dim)
        assert isinstance(self.domain, Iterable)
        self.domain = domain

    def __iter__(self):
        """Iterator for feasible domain."""
        for sol in self.domain:
            assert isinstance(sol, tuple)
            yield sol


class Objectives:
    """Represents the objectives of a multi-criteria optimisation problem.
    
    :ivar list objectives: Objective functions
    """

    def __init__(self):
        """Initialise with no objectives."""
        self._objectives = []

    def length(self):
        if self._objectives:
            for i, j in combinations(self._objectives, 2):
                assert len(i) == len(j)
            return len(self._objectives[0])
        else:
            return 0

    @property
    def obj(self):
        """Objective functions

        :getter: Returns objectives as list of tuples
        :setter: Appends objective (given by an Iterable) to list
        
        :Example: mult_crit.obj = [3, 2, -1] where mult_crit = Objectives()
        """
        return self._objectives

    @obj.setter
    def obj(self, objective):
        assert isinstance(objective, Iterable)
        self._objectives.append(tuple(objective))

    def __call__(self, sol):
        """Point in objective space corresponding to given solution.

        :param Iterable sol: Feasible solution
        :return: Image in objective space
        :rtype: tuple
        """
        return tuple([dot(array(obj), array(sol)) for obj in self._objectives])

    def __str__(self):
        """String representation of objectives."""
        return "\n".join(["obj_" + str(ind) + ": " + str(obj) for ind, obj
                          in enumerate(self._objectives, 1)])

    @classmethod
    def read(cls, file, *, delimiter=' '):
        """Read objectives given via file.

        :param str file: File(name) containing objectives
        :param str delimiter: Delimiter (Defaults to ' ')
        :rtype: :class:`Objectives`
        :raises FileNotFoundError: if file cannot be found
        :raise PermissionError: if file cannot be opened

        :Example: obj = Objectives.read(filename, delimiter=',') where \
                        filename corresponds to the location of the input file
        """
        assert isinstance(file, str)
        ins = cls()
        try:
            with open(file, mode='r', encoding='utf-8') as file:
                for line in file:
                    bracket_open = line.find("[")
                    bracket_close = line.find("]")
                    if bracket_open == -1 or bracket_close == -1:
                        continue
                    else:
                        substring = line[bracket_open+1:bracket_close]
                        ins._objectives.append(tuple(map(lambda p: int(p) if p.is_integer() else p, (float(p) for p in substring.split(sep=delimiter) if p))))
        except FileNotFoundError:
            print("File not found.")
        except PermissionError:
            print("Lacking permissions to open file.")
        else:
            return ins

class Points:
    """Represents certain set of points in objective space.

    :ivar str _id: Identifier for points
    :ivar str _color: Color used for visualisation of points
    :ivar list _visualised_items: Container for visualised items
    :ivar set points: Container for points
    """

    def __init__(self, identifier, color):
        """Initialises points and visualisation related items.

        :param str identifier: Identifier corresponding to point set
        :param str color: Color used for visualisation point set
        """
        self._id = identifier
        self._color = color
        self._visualised_items = []
        self.points = set()

    def __iter__(self):
        """Iterator for points."""
        iter(self.points)

    def __repr__(self):
        """Returns readable representation of points."""
        return str(self.points)

    def add(self, item):
        """Add item to points.

        :param tuple item: Point belonging to point set
        """
        self.points.add(item)

    def update(self, items):
        """Add items to points.

        :param iterable items: Points belonging to point set
        """
        self.points.update(items)

    def add_visualised_items(self, items):
        """Add items to container storing visualised objects.

        :param items: Visualised items
        :type items: :class:`matplotlib.collections.PathCollection`
        """
        self._visualised_items.append(items)

    def remove_visualised_items(self):
        """Remove all visualised items from matplotlib figure."""
        for elem in self._visualised_items:
            elem.remove()
        self._visualised_items = []

    @property
    def is_visualised(self):
        return self._visualised_items != []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, identifier):
        assert isinstance(identifier, str)
        self._id = identifier

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        assert isinstance(color, str)
        self._color = color

class PolyNondom:
    """Enumerates and visualises different sets of (non-dominated) points.
    
    .. glossary::
    
    non-dominated point
       A point *y* is non-dominated (in the point set P) if there is no other 
       point *z* in P such that *z_i <= y_i* for all i with at least one 
       strict inequality.
    
    dominated point
       A point is *y* is dominated if *y* is not non-dominated.
        
    polynon-dominated point
       A non-dominated point *y* is polynon-dominated (w.r.t. the underlying multi-objective problem)
       if the projection of the point is also non-dominated for the given multi-objective problem where
       one of the objectives was neglected

    mononon-dominated point
       A non-dominated point *y* is mononon-dominated if *y* is not 
       polynon-dominated.
        
    See also the `mathematical definitions <https://opus4.kobv.de/opus4-zib/files/6128/report_16-55.pdf>`_

    :ivar int _dim: Dimension of objective space
    :ivar Figure _fig: Figure object of matplotlib
    :ivar Axes _ax: Axes object of matplotlib
    :ivar str _message: Info message
    :ivar dict points: Maps indentifier to :class:`Points`
    :ivar list obj_to_polynd_points: maps objective to polynon-dominated points
    :ivar list polynd_boxes: Container for feasible boxes given by \
          polynon-dominated points
    :ivar dict _ax_limits: Maps coordinate axes to axes limits used in visualisation
    :ivar dict _ax_setter: Maps coordinate axes to matplotlib functions used for \
          setting axes limits
    """

    def __init__(self):
        """Initialises different point sets and visualisation related items."""
        self._dim = 0
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._message = "String combined of the following letters expected:"
        self.points = {'d': Points('dominated', 'black'),
                       'n': Points('non-dominated', 'red'),
                       'p': Points('polynon-dominated', 'blue'),
                       'm': Points('mononon-dominated', 'brown')}
        self.obj_to_polynd_points = []
        self.polynd_boxes = []
        self._ax_limits = {'x': [],
                           'y': [],
                           'z': []}
        self._ax_setter = {'x': self._ax.set_xlim3d,
                           'y': self._ax.set_ylim3d,
                           'z': self._ax.set_zlim3d}
        self._feasible_points = {}

    def __str__(self):
        """Returns readable representation of different point sets."""
        output = "Points:\n"
        for subset in self.points.values():
            output += subset.id + ": " + str(subset) + "\n"
        return output

    #@staticmethod
    def _generate_all_feasible_points(self, domain, objectives):
        """Generator for feasible points in objective space.
        
        :param iterable domain: Feasible domain
        :param `Objectives` objectives: Multiple objectives
        :return: Image in objective space
        :rtype: tuple
        """
        assert isinstance(domain, Iterable)
        assert isinstance(objectives, Objectives)
        for sol in domain:
            self._feasible_points[objectives(sol)] = sol
            yield objectives(sol)

    @staticmethod
    def _is_dominated(r, s, except_obj_index=None):
        """Checks whether r is dominated by s.
        
        :param tuple r: feasible point
        :param tuple s: feasible point
        :param int except_obj_index: index to neglect (Default is None.) 
        :return: True if r is dominated by s; false, otherwise.
        :rtype: bool
        
        .. note::
           Dominance relation is based on minimisation. The objective 
           corresponding to `except_obj_index` is neglected in the dominance
           check.
        """
        assert len(r) == len(s)
        strict_less = False
        for index, (elem_r, elem_s) in enumerate(zip(r, s)):
            if except_obj_index is not None and index == except_obj_index:
                continue
            elif elem_s > elem_r:
                return False
            elif elem_s < elem_r:
                strict_less = True
        return strict_less

    @staticmethod
    def _is_nondominated(point, points, except_obj_index=None):
        """Checks if point is non-dominated w.r.t. points."""
        for other in points:
            if PolyNondom._is_dominated(point, other, except_obj_index):
                return False
        return True

    @classmethod
    def compute_points(cls, domain, objectives):
        """Computes points sets based on given feasible domain and objectives.
        
        Computes dominated, non-dominated, polynon-dominated and 
        mononon-dominated points in objective space based on given domain 
        and objectives.
        
        :param `GenericDomain` domain: Feasible domain
        :param `Objectives` objectives: Objective functions
        :rtype: :class:`PolyNondom`
        
        :Example: ins = PolyNondom.compute_points(feas_dom, mult_crit) where \
                  feas_dom corresponds to :class:`GenericDomain` and \
                  mult_crit corresponds to :class:`Objectives`
        """
        assert isinstance(domain, GenericDomain)
        assert isinstance(objectives, Objectives)
        ins = cls()
        ins._compute(domain, objectives)
        return ins

    @staticmethod
    def _read_line(line, separator):
        """Converts line string to point(s)."""
        assert isinstance(line, str)
        start_ind = 0
        while True:
            bracket_open=line.find("[", start_ind)
            bracket_close=line.find("]", start_ind)
            if bracket_open == -1 or bracket_close == -1:
                break
            else:
                substring = line[bracket_open+1:bracket_close]
                start_ind = bracket_close+1
                yield tuple(map(lambda p: int(p) if p.is_integer()
                                else p, (float(p) for p in
                                         substring.split(sep=separator) if p)))
                
    @classmethod
    def read_points(cls, file, *, delimiter=' ', all_nondom=False):
        """Read points given via file.
        
        Reads all points given via file, computes dominated, non-dominated,
        polynon-dominated and mononon-dominated points and returns 
        corresponding instance.
        
        :param str file: File(name) containing points in objective space
        :param str delimiter: Delimiter for point coordinates (Defaults to ' ')
        :param bool all_nondom: Specifier for all given points are \
         non-dominated (Defaults to False)
        :rtype: :class:`PolyNondom`
        :raises FileNotFoundError: if file cannot be found
        :raise PermissionError: if file cannot be opened
        
        :Example: ins = PolyNondom.read_points(filename, delimiter=',') where \
                        filename corresponds to the location of the input file
        """
        assert isinstance(file, str)
        points = [] 
        ins = cls()
        try:
            with open(file, mode='r', encoding='utf-8') as file:
                for line in file:
                    for point in cls._read_line(line, delimiter):
                        ins.points['n'].add(point) if all_nondom else \
                            points.append(point)
        except FileNotFoundError:
            print("File not found.")
        except PermissionError:
            print("Lacking permissions to open file.")
        else:
            if not all_nondom:
                assert points
                ins._dim = len(points[0])
                for point in points:
                    if cls._is_nondominated(point, points):
                        ins.points['n'].add(point)
                    else:
                        ins.points['d'].add(point)
            else:
                assert ins.points['n'].points
                ins._dim = len(next(iter(ins.points['n'])))
            ins._compute_polynondom_points()
            ins._compute_monodom_points()
            ins._compute_boxes()
            return ins

    def _compute(self, domain, objectives):
        """Computes various non-dominated point sets."""
        self._dim = len(objectives.obj)
        points = list(self._generate_all_feasible_points(domain, objectives))
        for point in points:
            if self._is_nondominated(point, points):
                self.points['n'].add(point)
            else:
                self.points['d'].add(point)
        self._compute_polynondom_points()
        self._compute_monodom_points()
        self._compute_boxes()

    @staticmethod
    def _get_box(points):
        """Computes feasible rectangular box.

        The rectangular box is defined by 
        \bigtimes_i=1^k [\max_{i \in [k] \setminus \{i\}} p_i^j, p_i^i)
        where k coincides with self._dim and p^j \in N_{-j}
        
        :param Iterable points: Box defining points
        :return: Feasible box
        :rtype tuple
        :raises `InfeasibleBoxError`: if box is infeasible
        """
        assert isinstance(points, Iterable)
        assert points
        dim = len(next(iter(points)))
        box = []
        for index, elem in enumerate(zip(*points)):
            lhs = max([elem[j] for j in range(dim) if j != index])
            rhs = elem[index]
            if lhs < rhs:
                box.append((lhs, rhs))
            else:
                raise InfeasibleBoxError(" ".join(str(p) for p in points))
        return tuple(box)

    def _compute_boxes(self):
        """Computes all feasible rect. boxes given by polynon-dom. points."""
        for prod in product(*self.obj_to_polynd_points):
            try:
                box = self._get_box(prod)
            except InfeasibleBoxError:
                logging.info("Infeasible box: %s", str(prod))
            else:
                self.polynd_boxes.append(box)

    def _generate_nd_points(self, points, except_obj_index=None):
        """Generates non-dominated points w.r.t. given points
        
        :param Iterable points: Feasible points in objective space
        :param int except_obj_index: If given, corresp. objective is not 
          considered in non-dominance examination (Defaults to all objectives 
          are considered)
        :return: non-dominated point
        :rtype: tuple
        """
        for point in points:
            if self._is_nondominated(point, points, except_obj_index):
                yield point

    def _compute_polynondom_points(self):
        """Computes polynon-dominated points."""
        for index in range(self._dim):
            polynd_points = set(self._generate_nd_points(self.points['n'].points, index))
            self.obj_to_polynd_points.append(polynd_points)
            self.points['p'].update(polynd_points)

    def _compute_monodom_points(self):
        """Computes mononon-dominated points."""
        self.points['m'].update(self.points['n'].points.difference(self.points['p'].points))

    def _update_ax_limits(self, identifier, coordinates):
        """Update axis limits."""
        assert identifier in ['x', 'y', 'z']
        min_val = min(coordinates) - 2
        max_val = max(coordinates) + 2
        if not self._ax_limits[identifier]:
            self._ax_limits[identifier] = [min_val, max_val]
        else:
            if min_val < self._ax_limits[identifier][0]:
                self._ax_limits[identifier][0] = min_val
            if max_val > self._ax_limits[identifier][1]:
                self._ax_limits[identifier][1] = max_val

    def set_labels(self, *, my_xlabel="r'$c^{\top}_1 x$'",
                   my_ylabel="r'$c^{\top}_2 x$'",
                   my_zlabel="r'$c^{\top}_3 x$'",
                   my_labelsize=8, my_fontsize=12,
                   my_style='sci'):
        """Initialise labels."""
        self._ax.tick_params(axis='x', labelsize=my_labelsize)
        plt.ticklabel_format(style=my_style, axis='x', scilimits=(0,0))
        self._ax.tick_params(axis='y', labelsize=my_labelsize)
        plt.ticklabel_format(style=my_style, axis='y', scilimits=(0,0))
        self._ax.set_xlabel(my_xlabel, fontsize=my_fontsize)
        self._ax.set_ylabel(my_ylabel, fontsize=my_fontsize)
        if self._dim == 3:
            self._ax.tick_params(axis='z', labelsize=my_labelsize)
            plt.ticklabel_format(style=my_style, axis='z', scilimits=(0,0))
            self._ax.set_zlabel(my_zlabel, fontsize=my_fontsize)
        else:
            self._ax.set_zticks([])
            self._ax.w_zaxis.line.set_visible(False)
        plt.pause(0.0001)

    def _visualise_points(self, id, *, color, marker, marker_size):
        """Visualises given points.

        Visualises given points and saves corresponding matplotlib objects 
        in _visualised map.
        """
        x_coords, y_coords, *list_of_z_coords = zip(*self.points[id].points)
        z_coords = [0] if not list_of_z_coords else list_of_z_coords[0]
        self._update_ax_limits('x', x_coords)
        self._update_ax_limits('y', y_coords)
        self._update_ax_limits('z', z_coords)
        self._ax.scatter(x_coords, y_coords, z_coords, zdir='z', c=color, color=color,
                         s=marker_size, marker=marker, depthshade=False)
        self._ax_setter['x'](*self._ax_limits['x'])
        self._ax_setter['y'](*self._ax_limits['y'])
        self._ax_setter['z'](*self._ax_limits['z'])
        plt.pause(0.0001)

    def _visualise_lines(self, id, *, color, width, style):
        """Visualises line projections corresponding to given points.

        Visualises line projections of given points and saves corresponding 
        matplotlib objects in _visualised map.        
        """
        min_x, _ = self._ax_limits['x']
        min_y, _ = self._ax_limits['y']
        min_z, _ = self._ax_limits['z']
        for x, y, z in self.points[id].points:
            self._ax.plot([x, x], [min_y, y],[min_z, min_z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style)
            self._ax.plot([min_x, x], [y, y], [min_z, min_z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style)
            self._ax.plot([x, x], [y, y],[min_z, z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style)
            plt.pause(0.0001)

    def visualise(self, id, *, my_color=None, my_width=0.8, my_style='--',
                  my_marker='o', my_marker_size=40, with_lines=True):
        """Visualises point sets corresponding to id.
        
        :param str id: identifier corresponding to (combinations of) \
                       'd'ominated, 'n'on-dominated, 'p'olynon-dominated, \
                       'm'ononon-dominated points
        :param str my_color: color to use for visualisation (Defaults are \
                             dominated points - black,\
                             non-domintated points - blue,\ 
                             polynon-dominated points - green,\
                             mononon-dominated points - brown
        :param float my_width: width of lines (Default is 0.8)
        :param str my_style: style used for lines (Default is '--')
        :param str my_marker: marker used for points (Default is 'o')
        :param int my_marker_size: size used for marker (Default is 40)
        :param bool with_lines: Specifier if lines should be visualised \
                                in the 3d case (Default is True)

        :Example: ins.visualise('dp', my_color='blue') 
        """
        if self._dim < 2 or self._dim > 3:
            print("Can only visualise 2- and 3-dimensional objective spaces.\n")
        elif not isinstance(id, str) or \
                any([item not in self.points.keys() for item in id]):
            print(self._message)
            print(" ".join(self.points.keys()))
        else:
            for item in id:
                if self.points[item].points:
                    color = self.points[item].color if my_color is None else my_color
                    self._visualise_points(item, color=color, marker=my_marker,
                                           marker_size=my_marker_size)
                    if self._dim == 3 and with_lines:
                        self._visualise_lines(item, color=color, width=my_width,
                                              style=my_style)
            if __name__ == '__main__':
                plt.show()
            else:
                plt.draw()

    def save_figure(self, output_name, *, dpi=400, elevation=30, azimuth=50):
        self._ax.view_init(elevation, azimuth)
        self._fig.savefig(output_name, dpi=dpi)

    def visualise_polynd_boxes(self):
        """Visualises all feasible boxes given by polynon-dominated points."""
        for box in self.polynd_boxes:
            self.visualise_box(*box)

    def visualise_box(self, interval1, interval2, interval3=None, *,
                      my_face_color='blue', my_alpha=0.2,
                      my_xlabel="r'$c^{\top}_1 x$'",
                      my_ylabel="r'$c^{\top}_2 x$'",
                      my_zlabel=r'$c^{\top}_3 x$',
                      my_label_size=8, my_font_size=12):
        """Visualises given rectangular box.
        
        :param tuple interval1: first interval in Cartesian product \
                                corresponding to box
        :param tuple interval2: second interval in Cartesian product \
                                corresponding to box
        :param tuple interval3: (in 3d case) third interval in Cartesian \
                                product corresponding to box
        :param str my_face_color: color used to visualise faces of box \
                                  (Default is blue)
        :param float my_alpha: alpha value used for coloring faces
        
        :Example: ins.visualise_box((3, 4), (5, 7), (3, 5), my_face_color='red')
        
        .. todo::
           Adjust size of figure to include box in any case.
        """
        if self._dim < 2 or self._dim > 3:
            print("Can only visualise 2- and 3-dimensional objective spaces.\n")
        else:
            assert isinstance(interval1, tuple) and isinstance(interval2, tuple)
            x = linspace(*interval1, 3)
            y = linspace(*interval2, 3)
            x_y = meshgrid(x, y, sparse=True)
            if not interval3:
                self._ax.plot_surface(*x_y, 0, facecolors=my_face_color,
                                      alpha=my_alpha)
            else:
                assert isinstance(interval3, tuple)
                z = linspace(*interval3, 3)
                x_z = meshgrid(x, z, sparse=True)
                y_z = meshgrid(y, z, sparse=True)
                for i in interval1:
                    self._ax.plot_surface(i, *y_z, facecolors=my_face_color, alpha=my_alpha)
                for i in interval2:
                    self._ax.plot_surface(x_z[0], i, x_z[1], facecolors=my_face_color, alpha=my_alpha)
                for i in interval3:
                    self._ax.plot_surface(*x_y, i, facecolors=my_face_color, alpha=my_alpha)
            plt.draw()

    def clear_visualisation(self):
        """Clear current visualisation."""
        self._ax.clear()


def get_cmd_line_parser():
    """Command line interface."""
    parser = ArgumentParser(description='Enumerates and visualises different \
                                         sets of non-dominated points')
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('-f', '--file', metavar="input_file", type=str,
                               help='Each line in the input file is assumed to contain \
                                     an opening bracket [ and a closing bracket ] which \
                                     contain the input values.', required=True)
    parent_parser.add_argument('--delim', default=' ', type=str, metavar='delimiter',
                               help='Delimiter used to delimited the input values within \
                                     the brackets (defaults to space).')
    parent_parser.add_argument('-v', '--visualise', type=str, default="",
                               help="specify which points to visualise: (d)ominated,\
                               (n)on-dominated, (p)olynon-dominated,\
                               (m)ononon-dominated",
                               nargs='+', dest='vis', choices=['d', 'n', 'm', 'p'])
    parent_parser.add_argument('-c', '--color', type=str, default=None,
                               help="Color used for point visualisation",
                               choices=['red', 'blue', 'black', 'brown', 'green'])
    parent_parser.add_argument('--noLines', default=False, action='store_true',
                               help='Specify that line projections of points should not be \
                                     displayed. This might improve the general overview in \
                                     3d visualisation if many points are involved.')
    subparsers = parser.add_subparsers(dest='command')
    point_parser = subparsers.add_parser('points', parents=[parent_parser],
                                         description='Read pre-computed points given in input file.')
    point_parser.add_argument('--allNondom', default=False, action='store_true',
                              help='Switch to indicate that all pre-computed points are known \
                                   to be non-dominated (defaults to False)')
    obj_parser = subparsers.add_parser('objs', parents=[parent_parser],
                                       description='Read objectives given in input file.')
    obj_group = obj_parser.add_mutually_exclusive_group(required=True)
    obj_group.add_argument('--assignment', action='store_true', default=False,
                           help='Specify that corresponding domain is an assignment problem.')
    obj_group.add_argument('--cube', action='store_true', default=False,
                           help='Specify that corresponding domain are the vertices of a cube.')
    return parser

if __name__ == '__main__':
    my_parser = get_cmd_line_parser()
    args = my_parser.parse_args()
    if args.command == 'objs':
        objs = Objectives.read(args.file, delimiter=args.delim)
        if args.assignment:
            dim = math.sqrt(objs.length())
            assert dim == int(dim)
            domain = AssignmentDomain(int(dim))
        else:
            domain = CubeDomain(objs.length())
        ins = PolyNondom.compute_points(domain, objs)
    else:
        ins = PolyNondom.read_points(args.file, delimiter=args.delim,
                                     all_nondom=args.allNondom)
    print(ins)
    if args.vis:
        ins.visualise("".join(args.vis), my_color=args.color,
                      with_lines=not args.noLines)
    for i,j in ins._feasible_points.items():
        print(i, j)

