#!/usr/bin/env python3

"""Module for enumerating and visualising non-dominated points.

This moddule comprises classes for different feasible domains, a class for 
multiple objectives and a class for enumerating and visualising 
different subsets of (non-dominated) points.

Available classes
==================

- :class:`GenericDomain` represents the feasible domain of a(n abstract) 
  generic optimisation problem.

- :class:`AssignmentDomain` represents the feasible domain of an 
  assignment problem.

- :class:`CubeDomain` represents the feasible domain given by the vertices 
  of a standard cube.

- :class:`ExplicitDomain` represents the feasible domain given by an 
  explicitly given set of feasible solutions.

- :class:`Objectives` represents the objectives of a multi-criteria 
  optimisation problem.

- :class:`PolyNondom` enumerates and visualises different sets of 
  (non-dominated) points. 


Command Line Arguments
=======================

.. note:: Not all features of the above classes are available through 
   the command line interface.

Positional arguments
---------------------

- *read* specifies that pre-computed points are to be read from a file

- *create* specifies that the feasible domain and corresponding objectives 
   are to be created 

To display full help information: ``$ polynondom.py (read|create) -h``

Cmd args for *read*
++++++++++++++++++++

- ``-v | --visualise {d,n,m,p} [{d,n,m,p} ...]`` specifies which points to 
   visualise: (d)ominated, (n)on-dominated, (p)olynon-dominated, 
   (m)ononon-dominated

- ``--color {red,blue,black,brown,green}`` specifies point color 
   for visualisation

- ``--lines`` specifies that line projections should be visualised 
   (defaults to False)

- ``-f|--file FILE`` specifies file containing points to be read 

- ``--delim delimiter`` specifies delimiter character (defaults to space)

- ``--allNondom`` specifies that all given points are non-dominated 
   (defaults to False)

Cmd args for *create*
++++++++++++++++++++++

- ``-v | --visualise {d,n,m,p} [{d,n,m,p} ...]`` specifies which points 
   to visualise: (d)ominated, (n)on-dominated, (p)olynon-dominated, 
   (m)ononon-dominated

- ``--color {red,blue,black,brown,green}`` specifies point color 
   for visualisation 

- ``--lines`` specifies that line projections should be visualised 
   (defaults to False)

- ``-n | --noObjs INTEGER`` specifies number of objectives
  
- ``--randBounds INTEGER INTEGER`` specifies lower and upper bound for 
   random objective coefficients

- ``-a | --assignment INTEGER`` specifies number of agents of considered 
   assignment domain

- ``-c | --cube INTEGER`` specifies dimension of cube of considered 
   cube domain

"""

from argparse import ArgumentParser
from collections import defaultdict, Iterable
from itertools import chain, permutations, product
import logging
from numpy import array, dot, linspace, meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from random import randint

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


class Points:
    """Represents certain set of points in objective space."""

    def __init__(self, id, color):
        self.points = set()
        self._id = id
        self._color = color
        self._visualised_items = []

    def __iter__(self):
        iter(self.points)

    def __repr__(self):
        return str(self.points)

    def add(self, item):
        self.points.add(item)

    def update(self, items):
        self.points.update(items)

    def add_visualised_items(self, items):
        self._visualised_items.append(items)

    def remove_visualised_items(self):
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
       A non-dominated point *y* is polynon-dominated if 
 
    mononon-dominated point
       A non-dominated point *y* is mononon-dominated if *y* is not 
       polynon-dominated.
        
    See also `mathematical definitions <https://opus4.kobv.de/opus4-zib/files/6128/report_16-55.pdf>`_

    :ivar int _dim: Dimension of objective space
    :ivar Figure _fig: Figure object of matplotlib
    :ivar Axes _ax: Axes object of matplotlib
    :ivar dict points: Maps indentifier to `Points`
    :ivar list obj_to_polynd_points: maps objective to polynon-dominated points
    :ivar set monond_points: mononon-dominated points
    :ivar set polynd_boxes: feasible boxes given by polynon-dominated points


    :ivar str _message: Info message 
    :ivar defaultdict _visualised: Keeps track of what is already visualised
    """

    def __init__(self):
        """Initialises point sets and visualisation related items."""
        self._dim = 0
        self._fig = None
        self._ax = None
        self.points = {'d': Points('dominated', 'black'),
                       'n': Points('non-dominated', 'red'),
                       'p': Points('polynon-dominated', 'blue'),
                       'm': Points('mononon-dominated', 'brown')}
        self._message = "String combined of the following letters expected:"
        self.obj_to_polynd_points = []
        self.polynd_boxes = []
        self._visualised_boxes = []

    def __str__(self):
        """Returns readable representation of different point sets."""
        output = "Points:\n"
        for subset in self.points.values():
            output += subset.id + ": " + str(subset) + "\n"
        return output

    @staticmethod
    def _generate_all_feasible_points(domain, objectives):
        """Generator for feasible points in objective space.
        
        :param Iterable domain: Feasible domain
        :param `Objectives` objectives: Multiple objectives
        :return: Image in objective space
        :rtype: tuple
        """
        assert isinstance(domain, Iterable)
        assert isinstance(objectives, Objectives)
        for sol in domain:
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
                self.polynd_boxes.add(box)

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

    @staticmethod
    def _compute_ax_limits(points):
        """Update axes limits based on given points.
        
        Axis limits for each dimension are given by tuple consisting of minimal 
        objective value minus 1 and maximal objective value plus 1, 
        respectively, w.r.t. given points.
         
        :param Iterable points: Feasible points
        :rtype: list
        :return: List of tuples
        """
        assert isinstance(points, Iterable)
        return list([(min(item)-1, max(item)+1) for item in zip(*points)])

    def _update_ax_limits(self, points):
        """Update axis limits."""
        limits = PolyNondom._compute_ax_limits(points)
        assert len(limits) >= 2
        self._ax.set_xlim3d(*limits[0])
        self._ax.set_ylim3d(*limits[1])
        if len(limits) == 3:
            self._ax.set_zlim3d(*limits[2])


    def _init_visualisation(self):
        """Initialise visualisation-related objects."""
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_xlim3d(-20, 20)
        self._ax.set_ylim3d(-20, 20)
        self._ax.tick_params(axis='x', labelsize=8)
        self._ax.tick_params(axis='y', labelsize=8)
        self._ax.set_xlabel(r'$c^{\top}_1 x$', fontsize=12)
        self._ax.set_ylabel(r'$c^{\top}_2 x$', fontsize=12)
        if self._dim == 3:
            self._ax.set_zlim3d(-20, 20)
            self._ax.tick_params(axis='z', labelsize=8)
            self._ax.set_zlabel(r'$c^{\top}_3 x$', fontsize=12)
        else:
            self._ax.set_zticks([])
            self._ax.w_zaxis.line.set_visible(False)

    def _visualise_points(self, id, *, color, marker, marker_size):
        """Visualises given points.

        Visualises given points and saves corresponding matplotlib objects 
        in _visualised map.
        """
        x, y, *z = zip(*self.points[id].points)
        if not z:
            z = [0]
        self.points[id].add_visualised_items(self._ax.scatter(x, y, *z, zdir='z',
                                                              c=color, color=color,
                                                              s=marker_size,
                                                              marker=marker,
                                                              depthshade=False))
        plt.pause(0.001)

    def _visualise_lines(self, id, *, color, width, style):
        """Visualises line projections corresponding to given points.

        Visualises line projections of given points and saves corresponding 
        matplotlib objects in _visualised map.        
        """
        min_x, _ = self._ax.get_xlim3d()
        min_y, _ = self._ax.get_ylim3d()
        min_z, _ = self._ax.get_zlim3d()
        for x, y, z in self.points[id].points:
            self.points[id].add_visualised_items(*self._ax.plot([x, x], [min_y, y],
                                                               [min_z, min_z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style))
            self.points[id].add_visualised_items(*self._ax.plot([min_x, x], [y, y],
                                                               [min_z, min_z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style))
            self.points[id].add_visualised_items(*self._ax.plot([x, x], [y, y],
                                                               [min_z, z],
                                                               color=color,
                                                               linewidth=width,
                                                               linestyle=style))
            plt.pause(0.001)

    def visualise(self, id, *, my_color=None, my_width=0.8, my_style='--',
                  my_marker='o', my_marker_size=40, with_lines=True, show=True):
        """Visualises point set corresponding to id.
        
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
            if not self._fig:
                self._init_visualisation()
            for item in id:
                if self.points[item].points:
                    color = self.points[item].color if my_color is None else my_color
                    self._visualise_points(item, color=color, marker=my_marker,
                                           marker_size=my_marker_size)
            self._update_ax_limits(set.union(*[self.points[key].points
                                               for key in self.points.keys() if
                                               self.points[key].is_visualised]))
            if self._dim == 3 and with_lines:
                for item in id:
                    if self.points[item].points:
                        color = self.points[item].color if my_color is None else my_color
                        self._visualise_lines(item, color=color, width=my_width,
                                              style=my_style)
            if show:
                if __name__ == '__main__':
                    plt.show()
                else:
                    plt.draw()

    def save_figure(self, output_name, *, dpi=400, elevation=30, azimuth=50):
        self._ax.view_init(elevation, azimuth)
        self._fig.savefig(output_name, dpi=dpi)

    def undo_visualise(self, id):
        """Remove visualisation objects corresponding to given identifier.
        
        :param str id: identifier (combination of ('d', 'n', 'm', 'p', 'b') \
                       corresponding to objects that shall be removed from \
                       current visualisation 
       
        :Example: ins.undo_visualize('dp')
        """
        if not isinstance(id, str) or \
                not all([self.points[item].is_visualised for item in id]):
            print(self._message)
            print(" ".join([key for key in self.points.keys() if self.points[key].is_visualised]))
        else:
            for item in id:
                self.points[item].remove_visualised_items()
            plt.draw()

    def undo_box_visualisation(self):
        """Removes visualised boxes."""
        for box in self._visualised_boxes:
            box.remove()
        self._visualised_boxes = []

    def visualise_polynd_boxes(self):
        """Visualises all feasible boxes given by polynon-dominated points."""
        for box in self.polynd_boxes:
            self.visualise_box(*box)

    def visualise_box(self, interval1, interval2, interval3=None, *,
                      my_face_color='blue', my_alpha=0.2):
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
            if not self._fig:
                self._init_visualisation()
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
                    self._visualised_boxes.append(self._ax.plot_surface(i, *y_z,
                                                                       facecolors=my_face_color,
                                                                       alpha=my_alpha))
                for i in interval2:
                    self._visualised_boxes.append(self._ax.plot_surface(x_z[0],
                                                                       i, x_z[1],
                                                                       facecolors=my_face_color,
                                                                       alpha=my_alpha))
                for i in interval3:
                    self._visualised_boxes.append(self._ax.plot_surface(*x_y, i,
                                                                       facecolors=my_face_color,
                                                                       alpha=my_alpha))
            plt.draw()



    def close_visualisation(self):
        """Closes current visualisation window and resets related elements."""
        for key in self.points.keys():
            if self.points[key].is_visualised:
                self.points[key].remove_visualised_items()
        plt.close(self._fig)
        self._fig = None
        self._ax = None

if __name__ == '__main__':
    parser = ArgumentParser(description="Enumerates and visualises different \
                                         sets of non-dominated points")
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('-v', '--visualise', type=str, default="",
                        help="specify which points to visualise: (d)ominated,\
                             (n)on-dominated, (p)olynon-dominated,\
                             (m)ononon-dominated",
                        nargs='+', dest='vis', choices=['d', 'n', 'm', 'p'])
    parent_parser.add_argument('--color', type=str, default=None,
                        help="specify color",
                        choices=['red', 'blue', 'black', 'brown', 'green'])
    parent_parser.add_argument('--lines', default=True, action='store_false',
                               help="switch to specify that line projections \
                                    should not be displayed (defaults to True)")
    subparsers = parser.add_subparsers(help='help for commands', dest='command')
    point_parser = subparsers.add_parser('read', help='read help',
                                         parents=[parent_parser])
    point_parser.add_argument('-f', '--file', help='file containing points',
                              metavar="FILE", type=str, dest='p_file',
                              required=True)
    point_parser.add_argument('--delim', default=' ', type=str,
                              help="specify delimiter character \
                              (defaults to space)", metavar='delimiter',
                              dest='p_delim')
    point_parser.add_argument('--allNondom', default=False,
                              help="switch to specify that all points are \
                                   non-dominated (defaults to False)",
                              action='store_false', dest='p_allnondom')
    obj_parser = subparsers.add_parser('create', help='create help',
                                       parents=[parent_parser])
    obj_parser.add_argument('-n', '--noObjs', required=True,
                            help='specify number of objectives',
                            dest='o_num', type=int, metavar='INTEGER')
    obj_parser.add_argument('--randBounds', type=int, dest='o_rand',
                            help='Lower and upper bound for random number',
                            metavar="INTEGER", required=False,
                            nargs=2, default=(-10, 10))
    obj_group = obj_parser.add_mutually_exclusive_group(required=True)
    obj_group.add_argument("-a", "--assignment", metavar='INTEGER', type=int,
                           help="Number of agents of considered \
                                assignment domain", dest='a_dim')
    obj_group.add_argument("-c", "--cube", metavar='INTEGER', type=int,
                           help="Dimension of cube of considered cube domain",
                           dest='c_dim')
    args = parser.parse_args()
    ins = None
    if args.command == 'create':
        if args.a_dim:
            domain = AssignmentDomain(args.a_dim)
        else:
            domain = CubeDomain(args.c_dim)
        objs = Objectives()
        for i in range(args.o_num):
            objs.obj = [randint(*args.o_rand) for i in range(domain.dim)]
        ins = PolyNondom.compute_points(domain, objs)
    elif args.command == 'read':
        ins = PolyNondom.read_points(args.p_file, delimiter=args.p_delim,
                                     all_nondom=args.p_allnondom)
    if ins and args.vis:
        ins.visualise("".join(args.vis), my_color=args.color,
                      with_lines=args.lines)


